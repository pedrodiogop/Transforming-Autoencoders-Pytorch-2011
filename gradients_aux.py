import matplotlib.pyplot as plt
import os
import numpy as np


##################### AUX FUNCTIONS MEAN EACH CAPSULE ########################

def Save_Mean_Gradients_by_capsule(capL, grad_flow_caps):
    for name, param in capL.named_parameters():
                if param.grad is not None and 'weight' in name:
                    parts = name.split('.')        # ['caps', '0', 'inp_rec', 'weight']
                    cap_name = f'caps.{parts[1]}' # 'caps.0'
                    layer_name = parts[2]          # 'inp_rec'

                    if cap_name not in grad_flow_caps:
                        grad_flow_caps[cap_name] = {}
                    if layer_name not in grad_flow_caps[cap_name]:
                        grad_flow_caps[cap_name][layer_name] = []

                    grad_flow_caps[cap_name][layer_name].append(
                        param.grad.abs().mean().item()
                    )

    return grad_flow_caps


def Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR):
    layer_types = ['inp_rec', 'rec_xy', 'rec_prob', 'xy_gen', 'gen_out']
    colors      = ['blue', 'orange', 'green', 'red', 'purple']
    num_caps    = len(grad_flow_caps) # 25

    fig, axes = plt.subplots(num_caps, 1, figsize=(12, num_caps * 2))
    fig.suptitle(f'Gradient Flow por Cápsula — Época {epoch+1}', fontsize=10)

    for idx, (cap_name, layers) in enumerate(sorted(grad_flow_caps.items(),
                                             key=lambda x: int(x[0].split('.')[1]))):
        ax = axes[idx]

        for layer_name, color in zip(layer_types, colors):
            if layer_name in layers:
                ax.plot(layers[layer_name],
                        label=layer_name,
                        color=color,
                        linewidth=0.8)
                
        ax.set_yscale('log')
        ax.set_title(cap_name, fontsize=8)
        ax.set_ylabel('Grad abs mean', fontsize=7)
        ax.axhline(1e-6, color='red', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

        if idx == 0:
            ax.legend(fontsize=6, ncol=5, loc='upper right')

    axes[-1].set_xlabel('Batch')
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/grad_flow_capsules_ep_{epoch+1:03d}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


##################### AUX FUNCTIONS MEAN LAYER #####################

def Save_Mean_Gradients_by_layer(capL, grad_flow_layers):
    layer_grads = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}
    
    for name, param in capL.named_parameters():
        if param.grad is not None and 'weight' in name:
            parts = name.split('.')   # ['caps', '0', 'inp_rec', 'weight'] 
            layer_name = parts[2]     # 'inp_rec'
            if layer_name in layer_grads:
                layer_grads[layer_name].append(param.grad.abs().mean().item())

    for layer_name, values in layer_grads.items():
        if values:
            grad_flow_layers[layer_name].append(sum(values) / len(values))
                    
    return grad_flow_layers

def Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR):
    colors = {
        'inp_rec':  'blue',
        'rec_xy':   'orange',
        'rec_prob': 'green',
        'xy_gen':   'red',
        'gen_out':  'purple'
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'Gradient Flow por Camada — Época {epoch+1}', fontsize=12)

    for layer_name, values in grad_flow_layers.items():
        ax.plot(values,
                label=layer_name,
                color=colors[layer_name],
                linewidth=0.8,
                alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Grad abs mean (log)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(1e-6, color='red', linestyle='--', linewidth=0.8, label='vanishing')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/grad_flow_layers_ep_{epoch+1:03d}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

##################### AUX FUNCTIONS NOT USED #####################

# You can adapt this code for save all gradients across each batch_size 
def Save_Gradients(capl, epoch, RESULTS_DIR_GRADIENTS):
    for name, param in capl.named_parameters():
        if param.grad is not None:
            with open(RESULTS_DIR_GRADIENTS, 'a') as f:
                f.write(f'Epoch {epoch+1}, Layer: {name}, Grad Mean: {param.grad.mean().item():.6f} | max: {param.grad.max().item():.6f}\n')