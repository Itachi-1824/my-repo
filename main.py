import subprocess
import sys
import os

# set the custom directory to /home/container/python-packages for user installs
os.environ["PYTHONUSERBASE"] = "/home/container/python-packages"

# add /home/container/python-packages/bin to the path
python_packages_bin = "/home/container/python-packages/bin"
os.environ["PATH"] += os.pathsep + python_packages_bin

def install_redbot():
    """install redbot using pip to /home/container/python-packages."""
    print("installing redbot...")

    try:
        # upgrade pip and install necessary dependencies into the custom directory
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "--user"])

        # install redbot in the custom directory
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "Red-DiscordBot"])
    except subprocess.CalledProcessError as e:
        print(f"error during installation: {e}")
        sys.exit(1)

def run_custom_command():
    """Wait for user input and run the specified command."""
    print("Redbot has been installed. Now you can define an instance name or run a command.")
    
    # Prompt user to define the instance name or any other command
    user_input = input("Enter the instance name or command to execute: ")

    try:
        # Execute the user-defined command
        subprocess.check_call(user_input.split())
    except subprocess.CalledProcessError as e:
        print(f"error during command execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_redbot()
    run_custom_command()