from typing import *
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from contextlib import contextmanager
import os

class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)


    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    @torch.no_grad()
    def run_and_save(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        
        self._saved_coords = coords
        self._saved_slat = slat
        self._saved_cond = cond
        
        return self.decode_slat(slat, formats)


    @contextmanager
    def inject_slat_strength_edit(self, cond_orig: dict, cond_edit: dict, strength: float):
        sampler = self.slat_sampler
        sampler._old_sample_once = sampler.sample_once
        from .samplers import FlowEulerSampler

        def _new_sample_once(self_s, model, x_t, t, t_prev, cond, **kwargs):
            from easydict import EasyDict as edict
            
            out_orig = sampler._old_sample_once(model, x_t, t, t_prev, cond_orig['cond'], **kwargs)
            out_edit = sampler._old_sample_once(model, x_t, t, t_prev, cond_edit['cond'], **kwargs)
            
            pred_x_prev = (1 - strength) * out_orig.pred_x_prev + strength * out_edit.pred_x_prev
            pred_x_0 = (1 - strength) * out_orig.pred_x_0 + strength * out_edit.pred_x_0
            return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        sampler.sample_once = _new_sample_once.__get__(sampler, type(sampler))
        yield
        sampler.sample_once = sampler._old_sample_once
        delattr(sampler, '_old_sample_once')



    @torch.no_grad()
    def run_edit_from_saved(
        self,
        prompt_edit: str,
        strength: float = 0.5,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        assert hasattr(self, '_saved_coords') and hasattr(self, '_saved_cond'), \
            "call run_and_save() first"
        
        cond_edit = self.get_cond([prompt_edit])
        torch.manual_seed(seed)
        with self.inject_slat_strength_edit(self._saved_cond, cond_edit, strength):
            slat = self.sample_slat(self._saved_cond, self._saved_coords, slat_sampler_params)
        
        return self.decode_slat(slat, formats)



    def sample_sparse_structure(
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

        return 
    


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
    def run(
        self,
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
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    @torch.no_grad()
    def run_to_generate_data(
        self,
        prompt: str,
        path : str , 
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
        cond = self.get_cond([prompt])
        torch.save(cond , path)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    

    @torch.no_grad()
    def run_at_t(
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
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        noise = torch.load(path)
        coords = self.sample_diff_t_sparse_structure(noise , cond, num_samples, sparse_structure_sampler_params )
        
        # PATH_SAVE = f"/mnt/data/srihari/my_TRELLIS/LOAD_COORDS/{str(sparse_structure_sampler_params['cfg_strength'])}/{path.split('/')[-1][:-3]}"
        # os.makedirs(PATH_SAVE , exist_ok = True)
        # torch.save(coords , f"{PATH_SAVE}/coords.pt")
        # return
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
        



    @torch.no_grad()
    def sample_slats_for_interp(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt1: str,
        prompt2: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
    ):
        cond1 = self.get_cond([prompt1])
        cond2 = self.get_cond([prompt2])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat1 = self.sample_slat(cond1, coords, slat_sampler_params)
        torch.manual_seed(seed)
        slat2 = self.sample_slat(cond2, coords, slat_sampler_params)
        return slat1, slat2

    @torch.no_grad()
    def decode_slat_interp(
        self,
        slat1,
        slat2,
        alpha: float = 0.5,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        slat_interp = slat1.replace(slat1.feats * alpha + (1 - alpha) * slat2.feats)
        return self.decode_slat(slat_interp, formats)



    @torch.no_grad()
    def my_run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt1: str,
        prompt2: str,
        num_interpolations: int = 10,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ):
        coords = self.voxelize(mesh)
        if coords.shape[0] == 0:
            raise ValueError("Voxelization produced empty coords.")

        coords = torch.cat([
            torch.zeros(coords.shape[0], 1).int().cuda(),
            coords
        ], dim=1)

        # 2. Get endpoint conditionings
        cond1 = self.get_cond([prompt1])
        cond2 = self.get_cond([prompt2])



        slat1 = self.sample_slat(cond1, coords, slat_sampler_params)


        slat2 = self.sample_slat(cond2, coords, slat_sampler_params)

        results = []
        alphas = torch.linspace(0, 1, num_interpolations)

        for alpha in alphas:
            alpha = alpha.item()



            interp_slat = (1 - alpha) * slat1 + alpha * slat2
            obj = self.decode_slat(interp_slat, formats)
            results.append(obj)

        return results
    


  
    @torch.no_grad()
    def my_run_variant_slerp(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt1: str,
        prompt2: str,
        num_interpolations: int = 10,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ):
                
        def slerp(slat1, slat2, alpha):
            f1 = slat1.feats  # [V, C]
            f2 = slat2.feats  # [V, C]
            
            # Normalize
            f1_norm = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-8)
            f2_norm = f2 / (f2.norm(dim=-1, keepdim=True) + 1e-8)
            
            dot = (f1_norm * f2_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
            theta = torch.acos(dot)  # [V, 1]
            
            sin_theta = torch.sin(theta)
            
            # Where sin_theta is too small, fall back to lerp
            use_lerp = sin_theta.abs() < 1e-6
            
            w1 = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-8)
            w2 = torch.sin(alpha * theta) / (sin_theta + 1e-8)
            
            slerp_out = w1 * f1 + w2 * f2
            lerp_out  = (1 - alpha) * f1 + alpha * f2
            
            out = torch.where(use_lerp, lerp_out, slerp_out)
            
            # Sanity check
            print(f"NaN count: {out.isnan().sum()}, min: {out.min():.3f}, max: {out.max():.3f}")
            
            import trellis.modules.sparse as sp
            return sp.SparseTensor(feats=out, coords=slat1.coords)
            
        coords = self.voxelize(mesh)
        if coords.shape[0] == 0:
            raise ValueError("Voxelization produced empty coords.")

        coords = torch.cat([
            torch.zeros(coords.shape[0], 1).int().cuda(),
            coords
        ], dim=1)

        # 2. Get endpoint conditionings
        cond1 = self.get_cond([prompt1])
        cond2 = self.get_cond([prompt2])



        slat1 = self.sample_slat(cond1, coords, slat_sampler_params)


        slat2 = self.sample_slat(cond2, coords, slat_sampler_params)

        results = []
        alphas = torch.linspace(0, 1, num_interpolations)

        for alpha in alphas:
            alpha = alpha.item()



            interp_slat = slerp(slat1 , slat2 , alpha)
            obj = self.decode_slat(interp_slat, formats)
            results.append(obj)

        return results