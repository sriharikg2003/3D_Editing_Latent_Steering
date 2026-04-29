from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import open3d as o3d

import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp

import os
class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }


    # def sample_sparse_structure(
    #     self,
    #     cond: dict,
    #     num_samples: int = 1,
    #     sampler_params: dict = {},
    # ) -> torch.Tensor:
    #     """
    #     Sample sparse structures with the given conditioning.
        
    #     Args:
    #         cond (dict): The conditioning information.
    #         num_samples (int): The number of samples to generate.
    #         sampler_params (dict): Additional parameters for the sampler.
    #     """
    #     # Sample occupancy latent
    #     flow_model = self.models['sparse_structure_flow_model']
    #     reso = flow_model.resolution
    #     noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
    #     sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
    #     # sparse structure latent
    #     z_s = self.sparse_structure_sampler.sample(
    #         flow_model,
    #         noise,
    #         **cond,
    #         **sampler_params,
    #         verbose=True
    #     ).samples
    #     # breakpoint()
    #     # torch.save(z_s , 'intermediate.pt')

    #     # Decode occupancy latent
    #     decoder = self.models['sparse_structure_decoder']
    #     coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

    #     return coords


    # def sample_sparse_structure(
    #     self,
    #     cond: dict,
    #     num_samples: int = 1,
    #     sampler_params: dict = {},
    # ) -> torch.Tensor:
    #     """
    #     Sample sparse structures with the given conditioning.
    #     """
    #     flow_model = self.models['sparse_structure_flow_model']
    #     reso = flow_model.resolution
        
    #     # --- NOISE PERSISTENCE LOGIC ---
    #     noise_path = '/mnt/data/srihari/my_TRELLIS/LOAD/initial_noise.pt'
    #     if sampler_params.get('mode') == 'pass2' and os.path.exists(noise_path):
    #         # Load the exact noise from Pass 1
    #         noise = torch.load(noise_path, map_location=self.device)
    #     else:
    #         # Generate new noise for Pass 1
    #         noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
    #     # ------------------------------

    #     sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
    #     # Sample latent (Extract .samples from the EasyDict)
    #     z_s = self.sparse_structure_sampler.sample(
    #         flow_model,
    #         noise,
    #         **cond,
    #         **sampler_params,
    #         verbose=True
    #     ).samples

    #     # Decode occupancy latent
    #     decoder = self.models['sparse_structure_decoder']
    #     coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

    #     return coords

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        
        # 1. Noise persistence Logic
        noise_path = '/mnt/data/srihari/my_TRELLIS/LOAD/initial_noise.pt'
        if sampler_params.get('mode') == 'pass2' and os.path.exists(noise_path):
            noise = torch.load(noise_path, map_location=self.device, weights_only=True)
        else:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        # 2. This returns the edict from flow_euler.py
        res = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        
        # 3. Extract the tensor from the dictionary
        z_s = res.samples

        # 4. Decode to occupancy and extract coordinates
        decoder = self.models['sparse_structure_decoder']
        occupancy = decoder(z_s)
        
        # 5. This creates the [N, 4] Tensor that sample_slat needs
        coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()

        return coords

    @torch.no_grad()
    def invert_sparse_structure(
        self,
        cond: dict,
        z_s: torch.Tensor,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """DDIM-invert z_s → noise, caching latents per block per step."""
        flow_model = self.models['sparse_structure_flow_model']
        params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_T = self.sparse_structure_sampler.invert(
            flow_model, z_s, **cond, **params, verbose=True,
        )
        return z_T

    def sample_diff_t_sparse_structure(
        self,
        noise,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},

    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        # noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords
    
    
    def sample_sparse_scaled_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        # sparse structure latent
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples



        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords




    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat


    @torch.no_grad()
    def run_directional_edit(
        self,
        image_base: Image.Image,
        image_plus: Image.Image,
        image_minus: Image.Image,
        alpha: float = 1.0,
        num_samples: int = 1,
        seed: int = 42,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        if preprocess_image:
            image_base = self.preprocess_image(image_base)
            image_plus = self.preprocess_image(image_plus)
            image_minus = self.preprocess_image(image_minus)

        cond_base = self.get_cond([image_base])
        cond_plus = self.get_cond([image_plus])
        cond_minus = self.get_cond([image_minus])

        torch.manual_seed(seed)
        coords_base = self.sample_sparse_structure(cond_base, num_samples)
        slat_base = self.sample_slat(cond_base, coords_base)
        
        slat_plus = self.sample_slat(cond_plus, self.sample_sparse_structure(cond_plus, num_samples))
        slat_minus = self.sample_slat(cond_minus, self.sample_sparse_structure(cond_minus, num_samples))

        def get_flat_coords(s):
            return (s.coords[:, 1:].float() @ torch.tensor([4096, 64, 1], device=s.coords.device, dtype=torch.float)).to(torch.int64)

        c_base_flat = get_flat_coords(slat_base)
        c_plus_flat = get_flat_coords(slat_plus)
        c_minus_flat = get_flat_coords(slat_minus)

        intersect_mask = torch.isin(c_plus_flat, c_minus_flat)
        if intersect_mask.any():
            idx_plus = torch.where(intersect_mask)[0]
            idx_minus = torch.where(torch.isin(c_minus_flat, c_plus_flat))[0]
            
            direction = (slat_plus.feats[idx_plus] - slat_minus.feats[idx_minus]).mean(dim=0)
            slat_base.feats += alpha * direction
        else:
            print("Warning: No voxel intersection found for direction. Applying global mean shift.")
            direction = slat_plus.feats.mean(0) - slat_minus.feats.mean(0)
            slat_base.feats += alpha * direction

        return self.decode_slat(slat_base, formats)
 
    # @torch.no_grad()
    # def run_load_dino_vec(
    #     self,
    #     image: Image.Image,
    #     num_samples: int = 1,
    #     seed: int = 42,
    #     sparse_structure_sampler_params: dict = {},
    #     slat_sampler_params: dict = {},
    #     formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    #     preprocess_image: bool = True,
    #     alpha: float = 0.0,
    # ) -> dict:
    #     if preprocess_image:
    #         image = self.preprocess_image(image)
        
    #     cond_orig = self.get_cond([image])
        
    #     cond_steered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cond_orig.items()}
        
    #     if alpha != 0:
    #         thick_path = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS/thick_cylinders/thick_cylinders_04.png"
    #         thin_path = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS/thin_cylinders/thin_cylinders_11.png"
            
    #         img_thick = Image.open(thick_path)
    #         img_thin = Image.open(thin_path)
            
    #         cond_thick = self.get_cond([self.preprocess_image(img_thick)])
    #         cond_thin = self.get_cond([self.preprocess_image(img_thin)])
            
    #         diff_vec = cond_thick['cond'] - cond_thin['cond']
            
    #         orig_norms = torch.norm(cond_steered['cond'], dim=-1, keepdim=True)
            
    #         cond_steered['cond'] = cond_steered['cond'] + (alpha * diff_vec)
            
    #         new_norms = torch.norm(cond_steered['cond'], dim=-1, keepdim=True)
    #         cond_steered['cond'] = cond_steered['cond'] * (orig_norms / (new_norms + 1e-8))
            
    #         print(f"Applied Global Paired Difference Steering to Structure Sampler only (Alpha: {alpha})")

    #     torch.manual_seed(seed)
        
    #     coords = self.sample_sparse_structure(cond_steered, num_samples, sparse_structure_sampler_params)
        
    #     slat = self.sample_slat(cond_orig, coords, slat_sampler_params)

    #     return self.decode_slat(slat, formats)


    @torch.no_grad()
    def run_load_dino_vec(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        alpha: float = 0.0,
        crop_size: int = 10
    ) -> dict:
        if preprocess_image:
            image = self.preprocess_image(image)
        
        cond_orig = self.get_cond([image])
        
        cond_steered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cond_orig.items()}
        
        if alpha != 0:
            thick_path = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS/thick_cylinders/thick_cylinders_04.png"
            thin_path = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS/thin_cylinders/thin_cylinders_11.png"
            
            img_thick = Image.open(thick_path)
            img_thin = Image.open(thin_path)
            
            cond_thick = self.get_cond([self.preprocess_image(img_thick)])
            cond_thin = self.get_cond([self.preprocess_image(img_thin)])
            
            diff_vec = cond_thick['cond'] - cond_thin['cond']
            
            grid_size = 37
            mask_2d = torch.zeros((grid_size, grid_size), device=cond_steered['cond'].device)
            start = (grid_size - crop_size) // 2
            end = start + crop_size
            mask_2d[start:end, start:end] = 1.0
            
            spatial_mask = torch.ones((1, 1374, 1), device=cond_steered['cond'].device)
            spatial_mask[:, 5:, :] = mask_2d.view(1, -1, 1)
            
            orig_norms = torch.norm(cond_steered['cond'], dim=-1, keepdim=True)
            
            cond_steered['cond'] = cond_steered['cond'] + (alpha * diff_vec * spatial_mask)
            
            new_norms = torch.norm(cond_steered['cond'], dim=-1, keepdim=True)
            cond_steered['cond'] = cond_steered['cond'] * (orig_norms / (new_norms + 1e-8))
            
            print(f"Applied {crop_size}x{crop_size} Spatial Steering to Sparse Structure Sampler (Alpha: {alpha})")

        torch.manual_seed(seed)
        
        coords = self.sample_sparse_structure(cond_steered, num_samples, sparse_structure_sampler_params)
        
        slat = self.sample_slat(cond_orig, coords, slat_sampler_params)

        return self.decode_slat(slat, formats)



    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        # breakpoint()
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        # breakpoint()
        return self.decode_slat(slat, formats)


# 

        # 
        # os.makedirs(PATH_SAVE , exist_ok = True)
        # torch.save(coords , f"{PATH_SAVE}/coords.pt")
        # return
    



    @torch.no_grad()
    def run_at_t_load_cond_and_latent(
        self,
        path,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = torch.load('/mnt/data/srihari/my_TRELLIS/image_cond.pt')

        torch.manual_seed(seed)
        coords = torch.load(path)
        # coords = self.sample_diff_t_sparse_structure(noise , cond, num_samples, sparse_structure_sampler_params )
        



        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    

    @torch.no_grad()
    def run_move_in_latent(
        self,
        my_noise , 
        slat,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        alpha : float = 0.0,

    ) :




        new_feats = slat.feats + alpha * my_noise
        slat = slat.replace(new_feats)
        return self.decode_slat(slat, formats)
    

    @torch.no_grad()
    def sample_slat_and_noise(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
    ):


        image = self.preprocess_image(image)
        cond = self.get_cond([image])

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        noise = torch.randn(slat.feats.shape[1], device=slat.feats.device)
        noise = noise / noise.norm()
        return slat , noise



    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)



    @torch.no_grad()
    def run_sparse_interp_with_direction(
        self,
        image1: Image.Image,
        image2: Image.Image,
        actual_image : Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        alpha : float = 1,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        if preprocess_image:
            image1 = self.preprocess_image(image1)
            image2 = self.preprocess_image(image2)
            actual_image = self.preprocess_image(actual_image)
        
        cond1 = self.get_cond([image1])
        cond2 = self.get_cond([image2])
    
        cond_actual = self.get_cond([actual_image])

       
        raw_direction = cond2['cond'] - cond1['cond']


        actual_norm = torch.norm(cond_actual['cond'], dim=-1, keepdim=True)
        dir_norm = torch.norm(raw_direction, dim=-1, keepdim=True) + 1e-6
        normalized_direction = raw_direction * (actual_norm / dir_norm)

        cond_actual['cond'] = cond_actual['cond'] + alpha * normalized_direction

        cond_actual['cond'] = cond_actual['cond'] * (actual_norm / (torch.norm(cond_actual['cond'], dim=-1, keepdim=True) + 1e-6))

        coords = self.sample_sparse_structure(cond_actual, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat( self.get_cond([actual_image]), coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    


    @torch.no_grad()
    def run_edit_sparse(
        self,
        image1: Image.Image,
        image2: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        alpha : float = 1,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        if preprocess_image:
            image1 = self.preprocess_image(image1)
            image2 = self.preprocess_image(image2)


        
        cond1 = self.get_cond([image1])
        cond2 = self.get_cond([image2])


       


        # cond1['cond'] = cond1['cond']* alpha + cond2['cond']*(1-alpha)


        import torch




        def slerp(v0, v1, alpha, dot_threshold=0.9995):
            v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
            v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
            dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
            
            dot_clamped = dot.clamp(-1, 1)
            theta_0 = torch.acos(dot_clamped)
            sin_theta_0 = torch.sin(theta_0)
            
            theta_t = theta_0 * alpha
            sin_theta_t = torch.sin(theta_t)
            
            s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            
            res = s0 * v0 + s1 * v1
            
            linear = v0 * (1 - alpha) + v1 * alpha
            
            return torch.where(torch.abs(dot) > dot_threshold, linear, res)





        cond1['cond'] = slerp(cond1['cond'], cond2['cond'], alpha)

        

        coords = self.sample_sparse_structure(cond1, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat( cond1, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)


    @torch.no_grad()
    def sample_slats_for_interp(
        self,
        image1: Image.Image,
        image2: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
    ):

        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)

        cond1 = self.get_cond([image1])
        cond2 = self.get_cond([image2])
        coords1 = self.sample_sparse_scaled_structure(cond1, num_samples, slat_sampler_params )
        coords2 = self.sample_sparse_scaled_structure(cond2, num_samples, slat_sampler_params )

        torch.manual_seed(seed)
        slat1 = self.sample_slat(cond1, coords1, slat_sampler_params)
        torch.manual_seed(seed)
        slat2 = self.sample_slat(cond2, coords2, slat_sampler_params)
        return slat1, slat2


    @torch.no_grad()
    def sample_slat_one_image(
        self,
        image1: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
    ):

        image1 = self.preprocess_image(image1)

        cond1 = self.get_cond([image1])
        coords1 = self.sample_sparse_scaled_structure(cond1, num_samples, slat_sampler_params )

        torch.manual_seed(seed)
        slat1 = self.sample_slat(cond1, coords1, slat_sampler_params)


        return slat1




    @torch.no_grad()
    def run_sparse_scale(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,

    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_scaled_structure(cond, num_samples, sparse_structure_sampler_params )
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)