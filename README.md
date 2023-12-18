# RBE595_RL_Final_Project
Final Project for WPI course RBE 595ST. Reinforcement Learning Fall 2023
## Setting up the map
1. Copy contents of Map/ to folder **AirSim\Unreal\Environments\Blocks\Content\FlyingCPP\Maps**
2. Load the **Blocks** project.
## Using settings.json
1. Open **Blocks** project in AirSim
1. Press 'Play' button and click 'No' to select multirotor.
1. See messages that appear in simulation window. Look for one that says, "Loaded settings from </path/to/settings.json>"
1. Move **settings.json** from the Git repository to that directory and replace the file. 
1. Restart the sim.
## Running the model
The model can be run from the drone_sim.py script.
Inside the script there are three sections called **Testing**, **Debugging** and **Training**.
1. To see the drone fly using the most recently trained neural network, uncomment the **Testing** section
2. To train the drone, uncomment the **Training** section. 
## Other
1. Rewards for the training data are in **rewards_final.csv**.
2. Trained network is in **dqn.pt**.
