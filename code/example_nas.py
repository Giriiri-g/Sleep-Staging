"""
Example script for running Neural Architecture Search
This is a simplified example to demonstrate usage
"""
import torch
from torch.utils.data import DataLoader
from nas_search_space import Network, DARTS_OPS
from darts import DARTSTrainer
from rl_search import RLSearchTrainer, PolicyNetwork
from dataloader import LazyPSGDataset
from nas_evaluator import ArchitectureEvaluator, print_architecture

def example_darts():
    """Example of running DARTS search"""
    print("=" * 60)
    print("DARTS Example")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "sleep-edf-database-expanded-1.0.0/sleep-cassette"  # Update with your path
    
    # Load data
    dataset = LazyPSGDataset(folder_path=data_path, window_size=30)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = Network(
        input_channels=7,
        input_length=3000,
        num_classes=7,
        init_channels=32,  # Smaller for faster example
        num_cells=4,  # Fewer cells for faster example
        num_nodes=3,  # Fewer nodes for faster example
        search_space=DARTS_OPS
    )
    
    # Create trainer
    trainer = DARTSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        w_lr=0.025,
        alpha_lr=3e-4
    )
    
    # Train (fewer epochs for example)
    print("Training DARTS...")
    results = trainer.train(num_epochs=5, print_freq=1)
    
    # Get architecture
    arch = trainer.get_architecture()
    print("\nDiscovered Architecture:")
    print_architecture(arch)
    
    return results, arch


def example_rl():
    """Example of running RL-based search"""
    print("=" * 60)
    print("RL-based Search Example")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "sleep-edf-database-expanded-1.0.0/sleep-cassette"  # Update with your path
    
    # Load data
    dataset = LazyPSGDataset(folder_path=data_path, window_size=30)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create policy
    policy = PolicyNetwork(
        num_cells=4,  # Fewer cells for faster example
        num_nodes=3,  # Fewer nodes for faster example
        num_ops=14,  # Number of operations
        hidden_dim=64  # Smaller for faster example
    )
    
    # Create trainer
    trainer = RLSearchTrainer(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        reward_type='accuracy',
        temperature=1.0
    )
    
    # Search (fewer iterations for example)
    print("Searching with RL...")
    results = trainer.search(num_iterations=10, eval_epochs=2, print_freq=2)
    
    # Get best architecture
    arch = results['best_architecture']
    print("\nBest Architecture:")
    print_architecture(arch)
    
    return results, arch


if __name__ == "__main__":
    # Run DARTS example
    try:
        darts_results, darts_arch = example_darts()
        print(f"\nDARTS Best Val Acc: {darts_results['best_val_acc']:.2f}%")
    except Exception as e:
        print(f"DARTS example failed: {e}")
        print("Make sure to update data_path in the script")
    
    print("\n" + "=" * 60 + "\n")
    
    # Run RL example
    try:
        rl_results, rl_arch = example_rl()
        print(f"\nRL Best Reward: {rl_results['best_reward']:.2f}")
    except Exception as e:
        print(f"RL example failed: {e}")
        print("Make sure to update data_path in the script")

