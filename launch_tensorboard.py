#!/usr/bin/env python3
"""Launch TensorBoard to monitor CompactLoRa training."""

import subprocess
import sys
import webbrowser
import time
import os

def launch_tensorboard():
    """Launch TensorBoard and open in browser."""
    
    print("🚀 Launching TensorBoard for CompactLoRa training monitoring...")
    
    # Check if runs directory exists
    if not os.path.exists('runs'):
        print("❌ No 'runs' directory found. Start training first!")
        return
    
    try:
        # Launch TensorBoard
        print("📊 Starting TensorBoard server...")
        process = subprocess.Popen([
            sys.executable, '-m', 'tensorboard.main',
            '--logdir=runs',
            '--port=6006',
            '--reload_interval=1'  # Refresh every second
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open in browser
        url = "http://localhost:6006"
        print(f"🌐 Opening TensorBoard at {url}")
        webbrowser.open(url)
        
        print("\n✅ TensorBoard is running!")
        print("📈 You can now monitor:")
        print("   • Training progress in real-time")
        print("   • Reward curves")
        print("   • Compilation success rates")
        print("   • Generated code samples")
        print("\n💡 Press Ctrl+C to stop TensorBoard")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping TensorBoard...")
            process.terminate()
            
    except FileNotFoundError:
        print("❌ TensorBoard not installed. Install with:")
        print("   pip install tensorboard")
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")

if __name__ == "__main__":
    launch_tensorboard() 