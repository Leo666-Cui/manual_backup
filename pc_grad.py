import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random

def print_grad_stats(name, grad_tensor):
    if grad_tensor is None:
        print(f"Gradient Stats for '{name}': None")
        return
    
    # .detach() is important to avoid tracking this operation in the graph
    grad_tensor = grad_tensor.detach()
    
    # Check for non-finite numbers
    has_nan = torch.isnan(grad_tensor).any()
    has_inf = torch.isinf(grad_tensor).any()
    
    if has_nan or has_inf:
        print(f"!!! WARNING: Non-finite numbers found in '{name}' !!!")
        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
    else:
        # If the tensor is finite, print its stats
        print(f"Gradient Stats for '{name}':")
        print(f"    Shape: {grad_tensor.shape}")
        print(f"    Mean: {grad_tensor.mean():.6f}, Std: {grad_tensor.std():.6f}")
        print(f"    Min: {grad_tensor.min():.6f}, Max: {grad_tensor.max():.6f}")
        print(f"    Norm: {grad_tensor.norm():.6f}")

        
class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        # Use a more conservative epsilon to avoid numerical issues
        epsilon = 1e-6

        # Debug: Print initial gradient stats (reduced output)
        # print(f"\n=== PCGrad Projection Debug ===")
        # print(f"Shared parameters count: {shared.sum().item()}/{len(shared)}")
        # print(f"Non-shared parameters count: {(~shared).sum().item()}/{len(shared)}")
        
        for i, g in enumerate(grads):
            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"!!! WARNING: NaN/Inf in Input Grad {i} !!!")
            else:
                # print(f"Input Grad {i}: norm={g.norm().item():.6f}, mean={g.mean().item():.6f}")
                continue

        for i, g_i in enumerate(pc_grad):
            # Don't shuffle the original grads, use a copy for iteration order
            grad_indices = list(range(len(grads)))
            random.shuffle(grad_indices)
            
            for j_idx in grad_indices:
                if i == j_idx:
                    continue
                    
                g_j = grads[j_idx]  # Use original gradients for projection reference
                
                # Compute dot product and norms with better numerical stability
                g_i_g_j = torch.dot(g_i, g_j)
                g_j_norm_sq = torch.dot(g_j, g_j)  # More numerically stable than g_j.norm()**2
                
                # Only project if there's actual conflict and g_j has sufficient magnitude
                if g_i_g_j < 0 and g_j_norm_sq > epsilon:
                    # Compute projection coefficient
                    proj_coeff = g_i_g_j / (g_j_norm_sq + epsilon)
                    
                    # Clamp projection coefficient to prevent extreme projections
                    proj_coeff = torch.clamp(proj_coeff, min=-10.0, max=10.0)
                    
                    # Store original gradient for fallback
                    g_i_original = g_i.clone()
                    
                    # Apply projection (subtract the conflicting component)
                    g_i = g_i - proj_coeff * g_j
                    
                    # Check for numerical issues immediately after projection
                    if torch.isnan(g_i).any() or torch.isinf(g_i).any():
                        print(f"    !!! FATAL: NaN/Inf detected after projection for grad {i} !!!")
                        print(f"    - g_i norm before: {torch.dot(g_i_original, g_i_original).sqrt().item():.6f}")
                        print(f"    - g_j norm: {g_j_norm_sq.sqrt().item():.6f}")
                        print(f"    - Projection coefficient: {proj_coeff.item():.6f}")
                        # Fallback: use original gradient if projection failed
                        g_i = g_i_original
                        break
                    
                    # Check if gradient magnitude exploded (another safety check)
                    g_i_norm_after = torch.dot(g_i, g_i).sqrt()
                    g_i_norm_before = torch.dot(g_i_original, g_i_original).sqrt()
                    if g_i_norm_after > 10 * g_i_norm_before and g_i_norm_before > epsilon:
                        print(f"    !!! WARNING: Gradient magnitude exploded for grad {i} !!!")
                        print(f"    - Norm before: {g_i_norm_before.item():.6f}, after: {g_i_norm_after.item():.6f}")
                        # Use a scaled version to prevent explosion
                        g_i = g_i_original + 0.1 * (g_i - g_i_original)
                    
                    # Debug output
                    # print(f"Conflict found between grad {i} and {j_idx}. Dot product: {g_i_g_j:.6f}")
                    # print(f"    - g_j norm squared: {g_j_norm_sq.item():.6f}")
                    # print(f"    - Projection coefficient: {proj_coeff.item():.6f}")
                    # print(f"    - Gradient norm before: {g_i_norm_before.item():.6f}, after: {g_i_norm_after.item():.6f}")
                elif g_i_g_j < 0 and g_j_norm_sq <= epsilon:
                    print(f"Skipping projection for grad {i} vs {j_idx}: g_j norm too small ({g_j_norm_sq.item():.2e})")
            
            # Update the projected gradient
            pc_grad[i] = g_i

        # Debug: Print projected gradient stats (reduced output)
        for i, g in enumerate(pc_grad):
            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"!!! WARNING: NaN/Inf in Projected Grad {i} !!!")
            else:
                # print(f"Projected Grad {i}: norm={g.norm().item():.6f}, mean={g.mean().item():.6f}")
                continue

        # Merge gradients
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        
        if shared.any():  # Only process shared parameters if they exist
            if self._reduction == 'mean':
                merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
            elif self._reduction == 'sum':
                merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
            else: 
                raise ValueError('invalid reduction method')

        if (~shared).any():  # Only process non-shared parameters if they exist
            merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        
        # Debug: Print final merged gradient stats
        # print_grad_stats("Final Merged Grad", merged_grad)
        
        # Debug: Check the first few elements of merged gradient
        # print(f"First 10 elements of merged grad: {merged_grad[:10]}")
        # print(f"Elements around index 39425: {merged_grad[39420:39430]}")
        # print(f"Non-zero elements count: {(merged_grad != 0).sum().item()}")
        
        # print("=== End PCGrad Projection Debug ===\n")
        
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''
        # print(f"\n=== Setting Gradients Debug ===")
        # print(f"Total gradients to set: {len(grads)}")
        total_params = sum(1 for group in self._optim.param_groups for p in group['params'])
        trainable_params = sum(1 for group in self._optim.param_groups for p in group['params'] if p.requires_grad)
        # print(f"Total parameters in optimizer: {total_params}")
        # print(f"Trainable parameters: {trainable_params}")
        
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # Skip parameters that don't require gradients
                if not p.requires_grad:
                    continue
                    
                if idx >= len(grads):
                    print(f"Warning: Not enough gradients provided. Expected at least {idx+1}, got {len(grads)}")
                    break
                    
                grad_tensor = grads[idx]
                
                # Check for NaN/Inf before setting
                if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                    print(f"!!! WARNING: NaN/Inf in gradient for parameter {idx} !!!")
                    print(f"    Parameter shape: {p.shape}")
                    print(f"    Gradient shape: {grad_tensor.shape}")
                    # Set gradient to zero instead of NaN
                    grad_tensor = torch.zeros_like(p)
                
                # Handle data type conversion carefully
                if grad_tensor.dtype != p.dtype:
                    # Convert to parameter dtype, handling potential precision issues
                    if p.dtype == torch.float16 and grad_tensor.dtype == torch.float32:
                        grad_tensor = grad_tensor.to(torch.float16)
                    elif p.dtype == torch.float32 and grad_tensor.dtype in [torch.float16, torch.float64]:
                        grad_tensor = grad_tensor.to(torch.float32)
                    else:
                        grad_tensor = grad_tensor.to(p.dtype)
                
                # Ensure gradient has the same shape as parameter
                if grad_tensor.shape != p.shape:
                    print(f"!!! ERROR: Gradient shape mismatch for parameter {idx} !!!")
                    print(f"    Parameter shape: {p.shape}, Gradient shape: {grad_tensor.shape}")
                    # Skip this parameter to avoid crash
                    idx += 1
                    continue
                
                p.grad = grad_tensor
                
                # Debug: Print gradient stats for first few parameters
                if idx < 3:  # Only print for first 3 parameters to avoid spam
                    # print_grad_stats(f"Set Param {idx} Grad", grad_tensor)
                    # Also check if parameter itself has NaN after gradient is set
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print(f"!!! WARNING: Parameter {idx} contains NaN/Inf after gradient set !!!")
                
                idx += 1
        
        # print(f"=== End Setting Gradients Debug ===\n")
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for i, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            

            # --- 在这里添加打印 ---
            # Debug:
            # print(f"\n--- Checking raw gradients for Objective {i} (e.g., hcc_loss or pfs_loss) ---")
            # # 我们可以只看几个关键参数的梯度，比如第一个和最后一个
            # if grad:
            #     print_grad_stats(f"Objective {i} - First Param Grad", grad[0])
            #     print_grad_stats(f"Objective {i} - Last Param Grad", grad[-1])
            # --- 打印结束 ---


            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        # print(f"\n=== Unflatten Grad Debug ===")
        # print(f"Input grads shape: {grads.shape}")
        # print(f"Total shapes to unflatten: {len(shapes)}")
        
        unflatten_grad, idx = [], 0
        for i, shape in enumerate(shapes):
            length = int(np.prod(shape))
            if idx + length > len(grads):
                # print(f"!!! ERROR: Index out of bounds for shape {i} !!!")
                # print(f"    Shape: {shape}, Length: {length}")
                # print(f"    Current idx: {idx}, Required: {idx + length}, Available: {len(grads)}")
                # Use zeros as fallback
                unflatten_grad.append(torch.zeros(shape, device=grads.device, dtype=grads.dtype))
            else:
                grad_slice = grads[idx:idx + length].view(shape).clone()
                # Check for NaN in the slice
                if torch.isnan(grad_slice).any() or torch.isinf(grad_slice).any():
                    # print(f"!!! WARNING: NaN/Inf in unflatten grad slice {i} !!!")
                    # print(f"    Shape: {shape}, idx range: [{idx}:{idx + length}]")
                    pass  # Still add the gradient, but with warning
                unflatten_grad.append(grad_slice)
            idx += length
            
            # Debug first few parameters
            # if i < 3:
            #     print(f"Unflatten {i}: shape={shape}, length={length}, idx_range=[{idx-length}:{idx}]")
        
        # print(f"=== End Unflatten Grad Debug ===\n")
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        ** _retrieve_grad 只收集那些需要被训练的参数的梯度 **

        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # Skip parameters that don't require gradients
                if not p.requires_grad:
                    continue
                    
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                
                # Check for NaN/Inf in retrieved gradients
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"!!! WARNING: NaN/Inf detected in parameter gradient during retrieval !!!")
                    print(f"    Parameter shape: {p.shape}")
                    # Use zero gradient instead of NaN
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
