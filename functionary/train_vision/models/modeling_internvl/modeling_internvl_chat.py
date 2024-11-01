# This file was copied and modifed from https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel
from .modeling_internlm2 import InternLM2ForCausalLM
from .model_utility import (
    fill_image_tokens,
    load_pixel_values_from_image,
    get_aggregated_mask_after_truncation,
)
from PIL import Image

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True
    _no_split_modules = [
        "InternVisionModel",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
    ]

    def __init__(
        self, config: InternVLChatConfig, vision_model=None, language_model=None
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.36.2", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.tokenizer = None
        self.img_start_token = None
        self.img_end_token = None
        self.img_context_token = None
        self.img_place_holder_token = None

    def expand_input_ids(
        self,
        input_ids: torch.LongTensor,
        ori_labels: Optional[torch.LongTensor],
        attention_mask: torch.Tensor,
        images: List,
        training: bool = False,
    ):
        original_max_length = input_ids.shape[-1]
        bool_attention_mask = attention_mask.bool()
        batch_input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, bool_attention_mask)
        ]

        labels = ori_labels
        if labels is None:
            labels = torch.full_like(input_ids, -100)

        batch_labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, bool_attention_mask)
        ]

        imgs = [load_pixel_values_from_image(image, max_num=12) for image in images]

        pixel_values = torch.cat(imgs, dim=0).to(dtype=self.dtype, device=self.device)
        num_patches_list = [img.size(0) for img in imgs]
        index = 0

        new_batch_input_ids, new_batch_labels = [], []
        # We use a mask for masking image tokens that would be truncated by exceeding the max_length in the training
        # for example, we have 3 images in this batch: image1 (3 tokens, 1 was truncated); image2 (5 tokens, no tokens was truncated); image3 (4 tokens, 2 token was truncated)
        # aggregated_truncated_img_masks = [1, 1, 1, 0] + [1, 1, 1, 1, 1] + [1, 1, 0, 0]
        # we aggregate the mask because when we compute the representation of images, they would be aggregated
        aggregated_truncated_img_masks = []
        for c_input_ids, c_labels in zip(batch_input_ids, batch_labels):
            img_num = (c_input_ids == self.img_place_holder_token).sum()
            sub_num_patches_list = num_patches_list[index : index + img_num]
            index = index + img_num
            img_token_size_list = [
                num_patches * self.num_image_token
                for num_patches in sub_num_patches_list
            ]
            # print("img_token_size_list: ", img_token_size_list)
            new_input_ids, new_labels, img_masks = fill_image_tokens(
                c_input_ids.tolist(),
                c_labels.tolist(),
                img_token_size_list,
                self.img_start_token,
                self.img_context_token,
                self.img_end_token,
                self.img_place_holder_token,
            )
            # make sure that after filling image tokens, length won't exceed original_max_length
            if training and len(new_input_ids) > original_max_length:
                # print(f"truncate because input with image tokens exceeds: {len(new_input_ids)} > {original_max_length}")
                new_input_ids = new_input_ids[:original_max_length]
                new_labels = new_labels[:original_max_length]
                aggregated_truncated_img_masks.extend(
                    get_aggregated_mask_after_truncation(img_masks, original_max_length)
                )
            else:
                # no image tokens are truncated
                aggregated_truncated_img_masks.extend(
                    [1 for _ in range(sum(img_masks))]
                )

            new_batch_input_ids.append(new_input_ids)
            new_batch_labels.append(new_labels)

        # makse sure that all images have been processed
        assert index == len(num_patches_list)

        final_input_ids, final_labels, final_attention_masks = [], [], []
        for c_input_ids, c_labels in zip(new_batch_input_ids, new_batch_labels):
            pad_length = original_max_length - len(c_input_ids)
            final_input_ids.append(c_input_ids + [0 for _ in range(pad_length)])
            final_labels.append(c_labels + [-100 for _ in range(pad_length)])
            n_attention_mask = [1 for _ in range(len(c_input_ids))] + [
                0 for _ in range(pad_length)
            ]
            final_attention_masks.append(n_attention_mask)

        final_input_ids = torch.tensor(final_input_ids).to(device=input_ids.device)
        if ori_labels is not None:
            final_labels = torch.tensor(final_labels).to(device=ori_labels.device)
        else:
            final_labels = None

        final_attention_masks = torch.tensor(final_attention_masks).to(
            device=attention_mask.device
        )
        aggregated_truncated_img_masks = torch.tensor(aggregated_truncated_img_masks)
        assert (
            final_input_ids == self.img_context_token
        ).sum() == aggregated_truncated_img_masks.sum()
        return (
            final_input_ids,
            final_attention_masks,
            final_labels,
            pixel_values,
            aggregated_truncated_img_masks,
        )

    def forward(
        self,
        images: List,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # make sure that number of images == number of place holders
        assert len(images) == (input_ids == self.img_place_holder_token).sum()

        (
            input_ids,
            attention_mask,
            labels,
            pixel_values,
            aggregated_truncated_img_masks,
        ) = self.expand_input_ids(
            input_ids, labels, attention_mask, images, training=True
        )

        # print("aggregated_truncated_img_masks: ", aggregated_truncated_img_masks.tolist())

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        # print("input_ids shape: ", input_ids.shape)
        # print("input_ids: ", input_ids.tolist())
        # loss_count = (labels != -100).sum()
        # print("loss_count: ", loss_count)
        # print("attention_mask: ", attention_mask.sum())
        # print("part of pixel: ", pixel_values[0][0][0][: 10])
        vit_embeds = self.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.get_rank() == 0:
        #    print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token
        # print("number of image tokens: ", selected.sum())
        # print("shape of vit_embeds: ", vit_embeds.shape)
        vit_embeds = vit_embeds.reshape(-1, C)
        # print("after shaping of vit_embeds: ", vit_embeds.shape)
        # print("shape of: aggregated_truncated_img_masks: ", aggregated_truncated_img_masks.shape)
        vit_embeds = vit_embeds[
            aggregated_truncated_img_masks == 1
        ]  # remove image tokens that were truncated

        assert selected.sum() == vit_embeds.shape[0]
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
        # try:
        #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
        # except Exception as e:
        #     print(
        #         f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
        #         f"vit_embeds.shape={vit_embeds.shape}"
        #     )
        #     n_token = selected.sum()
        #     # Need to check this when batch_size_per_device > 1
        #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        result.logits = {"logits": logits, "labels": labels}
        return result

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                "Warning: `image_counts` is deprecated. Please use `num_patches_list` instead."
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        print("-------------QUERY")
        print(query)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        print("----------QUERY AFTER REPLACING IMAGE")
        print(query)
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
