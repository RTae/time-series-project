import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to sys.path
demo_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(demo_dir)
sys.path.insert(0, project_root)

from src.dataset import ThingsEEGDataset
from src.models.supaeeg import SUPAEEG
from src.encoders.vision_encoder import InternViTFeatureLookup
from src.utilities import Config, make_model

def load_config() -> Config:
    from omegaconf import OmegaConf
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(demo_dir)
    cfg = OmegaConf.load(os.path.join(project_root, "conf", "config.yaml"))
    config = Config()
    for field_name in config.__dataclass_fields__:
        if hasattr(cfg, field_name):
            setattr(config, field_name, getattr(cfg, field_name))
    # Make relative dataset/internvit paths absolute based on project_root
    if not os.path.isabs(config.dataset_dir):
        config.dataset_dir = os.path.abspath(os.path.join(project_root, config.dataset_dir))
    if not os.path.isabs(config.internvit_dir):
        config.internvit_dir = os.path.abspath(os.path.join(project_root, config.internvit_dir))
    return config

def main():
    parser = argparse.ArgumentParser(description="Generate a PNG image grid of retrieval results for slides.")
    parser.add_argument("--protocol", type=str, default="intra", choices=["intra", "inter"], help="Alignment protocol (intra or inter)")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-10)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to checkpoint (overrides --protocol/--subject default)")
    parser.add_argument("--output", type=str, default="retrieval_overview.png", help="Path to output PNG image")
    parser.add_argument("--rows", type=int, default=6, help="Number of rows (categories) to show")
    parser.add_argument("--categories", type=str, default="00001_aircraft_carrier,00002_antelope,00003_backscratcher,00011_batter,00017_boat,00018_bok_choy", help="Comma-separated concepts to display in order (overrides automatic selection)")
    parser.add_argument("--exclude", type=str, default="00017_boat,00007_basil", help="Comma-separated concepts to exclude")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config()
    
    # Determine checkpoint path based on protocol and subject
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        if args.protocol == "intra":
            checkpoint_path = os.path.join(demo_dir, "intra_full", "intra", f"supaeeg_intra_sub{args.subject:02d}.pt")
        else:
            checkpoint_path = os.path.join(demo_dir, "inter_full", "outputs", "2026-06-06", "inter", f"supaeeg_loso_sub{args.subject:02d}.pt")

    # Load test dataset (load_images=False to save memory/time; we load images manually on demand)
    dataset = ThingsEEGDataset(
        dataset_dir=config.dataset_dir,
        data_type="test",
        subject=args.subject,
        load_images=False,
        data_average=config.data_average_test
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # Retrieve all unique test concepts
    concepts = sorted(list(set(dataset.image_meta_data['test_img_concepts'])))
    concept_to_file = {}
    for i in range(len(dataset.image_meta_data['test_img_concepts'])):
        c = dataset.image_meta_data['test_img_concepts'][i]
        f = dataset.image_meta_data['test_img_files'][i]
        if c not in concept_to_file:
            concept_to_file[c] = f

    # Encode gallery image features
    feature_path = os.path.join(config.internvit_dir, "internvit_features.npy")
    lookup = InternViTFeatureLookup(feature_path=feature_path)
    files = [concept_to_file[c] for c in concepts]
    gallery_features = lookup.retrieve_batch(concepts, files)  # (200, 5, 3200)

    with torch.no_grad():
        zI = model.encode_image(gallery_features.to(device), subject_ids=None).cpu().numpy()  # (200, 512)

    # Perform retrieval across all 200 concepts to rank them
    all_results = []
    for concept_idx, target_concept in enumerate(concepts):
        # Find index in test dataset
        test_concept_idx = -1
        for i, c in enumerate(dataset.image_meta_data['test_img_concepts']):
            if c == target_concept:
                test_concept_idx = i
                break
        if test_concept_idx == -1:
            continue
            
        # Get indices for this concept's 80 trials
        indices = [test_concept_idx * dataset.number_of_repetitions + r for r in range(dataset.number_of_repetitions)]
        
        eeg_tensors = []
        target_file = None
        for idx in indices:
            eeg_tensor, _, _, _, _, _, img_file = dataset[idx]
            eeg_tensors.append(eeg_tensor)
            if target_file is None:
                target_file = img_file
                
        eeg_batch = torch.stack(eeg_tensors).to(device)
        
        with torch.no_grad():
            zE_trials = model.embed(eeg_batch)
            zE = torch.nn.functional.normalize(zE_trials.mean(dim=0, keepdim=True), dim=1).cpu().numpy()
            
        sim = cosine_similarity(zE, zI)[0]
        sorted_indices = np.argsort(-sim)
        target_rank = int(np.where(sorted_indices == concept_idx)[0][0]) + 1
        
        top_5_results = []
        for rank_idx, idx in enumerate(sorted_indices[:5], 1):
            top_5_results.append({
                "rank": rank_idx,
                "concept": concepts[idx],
                "image_file": concept_to_file[concepts[idx]],
                "similarity": float(sim[idx])
            })
            
        all_results.append({
            "gt_concept": target_concept,
            "gt_file": target_file,
            "rank": target_rank,
            "top_5": top_5_results
        })

    # Select the representative runs
    if args.categories:
        cat_list = [c.strip() for c in args.categories.split(",") if c.strip()]
        selected_runs = []
        for cat in cat_list:
            for r in all_results:
                if r['gt_concept'] == cat:
                    selected_runs.append(r)
                    break
        args.rows = len(selected_runs)
    else:
        # Fall back to automatic selection logic
        exclude_list = [c.strip() for c in args.exclude.split(",") if c.strip()]
        filtered_results = [r for r in all_results if r['gt_concept'] not in exclude_list]

        rank_1 = [r for r in filtered_results if r['rank'] == 1]
        rank_2_5 = [r for r in filtered_results if 2 <= r['rank'] <= 5]
        rank_gt_5 = [r for r in filtered_results if r['rank'] > 5]

        # Pick 50% Top-1, 30% Top-5, and remainder from lower ranks
        num_rank_1 = max(1, int(args.rows * 0.5))
        num_rank_2_5 = max(1, int(args.rows * 0.3))
        num_rank_gt_5 = max(1, args.rows - num_rank_1 - num_rank_2_5)

        selected_runs = []
        selected_runs.extend(rank_1[:min(num_rank_1, len(rank_1))])
        selected_runs.extend(rank_2_5[:min(num_rank_2_5, len(rank_2_5))])
        selected_runs.extend(rank_gt_5[:min(num_rank_gt_5, len(rank_gt_5))])

        # Fill up to requested rows if needed
        remaining_needed = args.rows - len(selected_runs)
        if remaining_needed > 0:
            seen = {r['gt_concept'] for r in selected_runs}
            for r in filtered_results:
                if r['gt_concept'] not in seen:
                    selected_runs.append(r)
                    seen.add(r['gt_concept'])
                    if len(selected_runs) == args.rows:
                        break

        # Sort selected runs by rank (ascending) so successes are displayed first
        selected_runs = sorted(selected_runs, key=lambda x: x['rank'])[:args.rows]

    # Create the high-resolution grid using Matplotlib
    print(f"Creating grid layout ({args.rows} rows x 6 columns) and saving to {args.output}...")
    num_rows = len(selected_runs)
    top_k = 5
    
    # Large figsize and high DPI to make the text crisp and readable in presentations
    fig, axes = plt.subplots(num_rows, top_k + 1, figsize=(15, 2.5 * num_rows), dpi=150)
    fig.patch.set_facecolor('#ffffff')  # White background for slide inclusion

    # Add Column Headers
    headers = ["GT", "Top1", "Top2", "Top3", "Top4", "Top5"]
    for col_idx, header in enumerate(headers):
        axes[0, col_idx].set_title(header, fontsize=16, pad=12, fontweight='bold', color='#2c3e50')

    # Draw each row
    for row_idx, run in enumerate(selected_runs):
        gt_concept = run["gt_concept"]
        gt_file = run["gt_file"]
        
        # 1. Plot Ground Truth (GT)
        gt_path = os.path.join(config.dataset_dir, "test_images", gt_concept, gt_file)
        ax = axes[row_idx, 0]
        
        try:
            img = Image.open(gt_path).convert('RGB')
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Image Error\n{gt_concept}", ha='center', va='center', fontsize=9)
            print(f"Error loading GT image {gt_path}: {e}")

        clean_gt_name = gt_concept.split('_', 1)[1] if '_' in gt_concept else gt_concept
        ax.set_xlabel(clean_gt_name, fontsize=11, fontweight='semibold', labelpad=4)
        
        # Hide ticks and draw a clean border around reference GT
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#3498db')  # Blue border for GT
            spine.set_linewidth(3)

        # 2. Plot top 5 retrieved candidates
        for col_idx, retrieved in enumerate(run["top_5"], 1):
            ret_concept = retrieved["concept"]
            ret_file = retrieved["image_file"]
            similarity = retrieved["similarity"]
            
            ret_path = os.path.join(config.dataset_dir, "test_images", ret_concept, ret_file)
            ax = axes[row_idx, col_idx]
            
            try:
                img = Image.open(ret_path).convert('RGB')
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Image Error\n{ret_concept}", ha='center', va='center', fontsize=9)
                print(f"Error loading retrieved image {ret_path}: {e}")

            clean_ret_name = ret_concept.split('_', 1)[1] if '_' in ret_concept else ret_concept
            ax.set_xlabel(f"{clean_ret_name}\n(sim: {similarity:.3f})", fontsize=10, labelpad=4)

            # Match highlight border check
            is_match = (ret_concept == gt_concept)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                if is_match:
                    spine.set_color('#2ecc71')  # Green border for success matches
                    spine.set_linewidth(3.5)
                    ax.xaxis.label.set_color('#27ae60')
                    ax.xaxis.label.set_weight('bold')
                else:
                    spine.set_color('#bdc3c7')  # Gray border for other candidates
                    spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Retrieval overview image successfully saved to {args.output}")

if __name__ == "__main__":
    main()
