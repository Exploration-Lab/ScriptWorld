import os
import json
import time
import argparse
import random
from typing import List, Dict, Any
from Script_World import ScriptWorldEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

SCENARIOS = [
    "repairing a flat bicycle tire",
    "borrowing a book from the library",
    "taking a bath",
    "going grocery shopping",
    "going on a train",
    "planting a tree",
    "flying in an airplane",
    "baking a cake",
    "riding on a bus",
    "getting a hair cut",
]

def load_hints(scenario: str) -> Dict[str, List[str]]:
    """Load hints for the given scenario from a JSON file."""
    with open(f"hints/{scenario}.json") as f:
        return json.load(f)

def print_colored(text: str, color: str = "\033[39m", end: str = "\n"):
    """Print colored text."""
    print(f"{color}{text}\033[39m", end=end)

def print_separator(char: str = "*", length: int = 100):
    """Print a separator line."""
    print_colored(char * length, "\033[0;35m")

def print_centered_text(text: str, total_width: int = 100, color: str = "\033[1;32m"):
    """Print centered text within separators."""
    padding = (total_width - len(text) - 2) // 2
    print_colored("*" * padding + " ", "\033[0;35m", end="")
    print_colored(text, color, end="")
    print_colored(" " + "*" * padding, "\033[0;35m")

def print_progress_bar(percentage: float):
    """Print a progress bar."""
    filled = int(percentage/100 * 90)
    empty = 90 - filled
    print_colored(f"{'█' * filled}{' ' * empty}| {percentage:.2f}%", "\033[0;36m")

def run_game(args: argparse.Namespace):
    """Run the Script World game."""
    env = ScriptWorldEnv(
        scenario=args.scenario,
        num_actions=args.num_actions,
        allowed_wrong_actions=args.allowed_wrong_actions,
        hop=args.hop,
        seed=args.seed,
        disclose_state_node=args.disclose_state_node,
    )
    
    hints = load_hints(args.scenario)
    life = args.allowed_wrong_actions
    total_reward = 0
    done = False

    while not done and life > 0:
        if not args.history:
            os.system('clear')
        print_separator()
        print_centered_text(text=env.quest.partition(".")[0])
        print_separator()

        if args.disclose_state_node:
            hint = random.choice(hints[str(env.state).replace("-", "_")])
            print_colored(f"HINT: {hint}", "\033[1;34m")
            print_separator()

        print_colored("ACTIONS:")
        for i, action in enumerate(env.action_spaces):
            print_colored(f"  {i}: {action}", "\033[1;33m")
        print_separator()

        while True:
            try:
                choice = int(input("Choose an Action: "))
                if 0 <= choice < len(env.action_spaces):
                    break
                raise ValueError()
            except ValueError:
                print_colored("Invalid Input!! Try Again.", "\033[0;31m")

        print_separator()
        action = env.action_spaces[choice]
        print_colored(f"Selected Choice: {action}")

        _, reward, done, _ = env.step(choice)
        total_reward += reward
        if reward < 0:
            life += reward

        print_colored(f"Point Acquired: {reward}")
        print_colored(f"Total reward: {total_reward}")
        print_colored(f"Lives Left: {life}")
        print_colored(f"Percentage completion: {env.completion_percentage:.2f}%")
        print_progress_bar(env.completion_percentage)
        print_separator()

        if reward >= 0:
            print_centered_text("Right Choice!")
        else:
            print_centered_text("WRONG CHOICE! 1 Life Deducted", color="\033[1;31m")
        print_separator()

        if not args.history:
            time.sleep(2)

    if life <= 0:
        print_centered_text("GAME OVER YOU LOST!!!", color="\033[1;31m")
        print_separator()
    elif done:
        print_centered_text("GAME OVER YOU WON!!")
        print_separator()

def main():
    parser = argparse.ArgumentParser(description="Script World Game Runner")
    parser.add_argument('--scenario', type=str, default='baking a cake', choices=SCENARIOS)
    parser.add_argument('--num_actions', type=int, default=5)
    parser.add_argument('--allowed_wrong_actions', type=int, default=5)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--disclose_state_node', action="store_true", help="shows state nodes/hints")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--history", action="store_true", help="Shows the history of previous steps")

    args = parser.parse_args()
    run_game(args)

if __name__ == "__main__":
    main()