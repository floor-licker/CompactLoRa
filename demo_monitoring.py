#!/usr/bin/env python3
"""Demo TensorBoard monitoring for CompactLoRa training."""

import time
import random
from torch.utils.tensorboard import SummaryWriter

def demo_training_monitoring():
    """Simulate training with TensorBoard logging."""
    
    print("🎬 Demo: TensorBoard Monitoring for CompactLoRa")
    print("📊 This will simulate training progress...")
    
    # Initialize TensorBoard
    writer = SummaryWriter('runs/demo_training')
    
    # Simulate 20 training iterations
    for iteration in range(20):
        # Simulate improving metrics over time
        base_reward = 0.5 + (iteration * 0.03) + random.uniform(-0.1, 0.1)
        success_rate = min(0.8, 0.1 + (iteration * 0.04) + random.uniform(-0.05, 0.05))
        complexity = 1.5 + (iteration * 0.08) + random.uniform(-0.2, 0.2)
        positive_examples = random.randint(0, min(10, iteration + 1))
        
        # Log metrics
        writer.add_scalar('Training/Average_Reward', base_reward, iteration)
        writer.add_scalar('Training/Success_Rate', success_rate, iteration)
        writer.add_scalar('Training/Average_Complexity', complexity, iteration)
        writer.add_scalar('Training/Positive_Examples', positive_examples, iteration)
        
        # Log sample generated code
        sample_codes = [
            f"pragma language_version >= 0.14.0;\\n\\nexport circuit Example_{iteration} {{\\n  // Complexity: {complexity:.1f}\\n}}",
            f"pragma language_version >= 0.14.0;\\n\\nexport ledger State_{iteration}: Uint<64>;\\n\\nexport circuit update(): [] {{\\n  State_{iteration} += 1;\\n}}",
            f"pragma language_version >= 0.14.0;\\n\\nmodule Advanced_{iteration} {{\\n  // Generated with reward: {base_reward:.3f}\\n}}"
        ]
        
        if positive_examples > 0:
            sample_code = random.choice(sample_codes)
            writer.add_text('Generated_Code/Sample', sample_code, iteration)
        
        print(f"Iteration {iteration+1:2d}: Reward={base_reward:.3f}, Success={success_rate:.1%}, Complexity={complexity:.1f}")
        time.sleep(0.5)  # Simulate training time
    
    writer.close()
    print("\n✅ Demo completed!")
    print("🚀 Launch TensorBoard with: python3 launch_tensorboard.py")
    print("📈 You'll see:")
    print("   • Reward curve trending upward")
    print("   • Success rate improving over time") 
    print("   • Complexity increasing")
    print("   • Generated code samples")

if __name__ == "__main__":
    demo_training_monitoring() 