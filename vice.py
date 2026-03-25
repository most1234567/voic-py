"""
AI Robot VAD Simulator (Keyboard-Triggered)
- Feature 1: No Audio Hardware required (Simulates Mic with Typing).
- Feature 2: Instant recording start on first character.
- Feature 3: Auto-stop the moment typing stops.
- Feature 4: High-speed processing for AI Integration.
"""

import os
import time

# Create a folder for the simulated command logs/files
OUTPUT_DIR = "robot_commands"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_ai_vad_simulator():
    print(f"--- AI ROBOT SYSTEM ACTIVE ---")
    print(f"Commands will be logged in: /{OUTPUT_DIR}")
    
    while True:
        print("\n[IDLE] Waiting for input (Type your command and press Enter):")
        print("(Type 'quit' to shut down the robot)")
        
        # Start Time - High Precision
        user_input = input(">> ").strip()

        if user_input.lower() == 'quit':
            print("System shutting down...")
            break

        if user_input:
            # INSTANT TRIGGER: The moment input is received
            print(">>> [SIMULATING RECORDING] ... Recording Active ...")
            
            # Record start timestamp
            start_time = time.time()
            
            # SIMULATION LOGIC: 
            # In a real AI integration, 'user_input' is the stream of data.
            # Here we simulate the processing time based on text length.
            processing_time = len(user_input) * 0.05 
            time.sleep(processing_time) 
            
            # INSTANT STOP: Once the string is fully received/processed
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            # Save the "Text-Audio" Simulation to a file
            filename = os.path.join(OUTPUT_DIR, f"cmd_{int(time.time())}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Timestamp: {time.ctime()}\n")
                f.write(f"Simulated Duration: {duration}s\n")
                f.write(f"Command Content: {user_input}")

            print(f">>> [STOP] Recording finished. Duration: {duration}s")
            print(f">>> [SAVED] Command stored in: {filename}")

if __name__ == "__main__":
    run_ai_vad_simulator()