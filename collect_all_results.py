import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, result_file):
    """
    Run the specified Python script and save the output to the result file
    """
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running script: {script_name}\n")
        f.write(f"{'='*80}\n\n")
        
        try:
            # Run script and capture output
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Write output in real-time
            for line in process.stdout:
                print(line, end='')
                f.write(line)
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                f.write(f"\n\nScript {script_name} executed successfully\n")
            else:
                f.write(f"\n\nScript {script_name} execution failed, exit code: {process.returncode}\n")
        
        except Exception as e:
            error_msg = f"Error occurred while running script {script_name}: {str(e)}\n"
            print(error_msg)
            f.write(error_msg)

def main():
    # Results file path
    result_file = 'all_models_results.txt'
    
    # Clear or create result file
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"MovieLens Recommendation System - All Model Results\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{'='*80}\n")
    
    print(f"Starting to run all models, results will be saved to: {result_file}")
    
    # Run Scikit-learn model test script
    run_script('test_scikit_model.py', result_file)
    
    # Run PyTorch model test script
    run_script('test_pytorch_model.py', result_file)
    
    # Run model comparison analysis script
    run_script('analyze_models.py', result_file)
    
    print(f"\nAll models execution completed, results have been saved to: {result_file}")

if __name__ == "__main__":
    main()