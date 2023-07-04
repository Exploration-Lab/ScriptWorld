from Script_World import ScriptWorldEnv
import random
import numpy as np
import time
import json
import pandas as pd
import random
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


print("\033[39m")
scenario = [
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


def run_game(args):
    scen = []
    asr = []
    sid = []
    cig = []
    sc = []
    ph = []
    cic = []
    cr = []
    gs = []
    gcp = []
    tim = []
    scn = args.scn
    cnt = 0
    no_of_actions = args.no_of_actions
    allowed_wrong_actions = args.allowed_wrong_actions
    life = args.allowed_wrong_actions
    hop = args.hop

    hi = "Y" if args.disclose_state_node else "N"
    dn = True if args.disclose_state_node else False

    mem = "Y" if args.history else "N"
    env = ScriptWorldEnv(
        scn=args.scn,
        no_of_actions=args.no_of_actions,
        allowed_wrong_actions=args.allowed_wrong_actions,
        hop=args.hop,
        seed=args.seed,
        disclose_state_node=args.disclose_state_node,

    )

    fname = args.scn + ".json"
    f = open(f"hints/{fname}")
    hnt = json.load(f)

    os.system('clear')

    R = []
    done = False
    len_episode = []
    reward = 0

    while not done and life > 0:

        scen.append(scn)

        if cnt != 0:

            print("  Point Acquired :", r)
            print("  Total reward : ", reward)
            print("  Lives Left : ", life)
            print("  Percentage completion:", round(
                env.per_clp * 100, 2), "%\n")
            print("\033[0;36m", u"█" * int(env.per_clp * 90)
                  + " " * round(int((1 - env.per_clp) * 90), 2)
                  + "| ",
                  round(env.per_clp * 100, 2),
                  "%\n",
                  )

        done = env.done
        print(
            "\033[0;35m ***************************************************************************************************"
        )
        scenario_display_string = env.quest.partition(".")[0]
        scenario_display_string = scenario_display_string if scenario_display_string.__len__() % 2 == 0 else scenario_display_string + " "
        print(
            "\033[0;35m",
            "*" * ((100-scenario_display_string.__len__())//2 -3),
            "\033[1;32m", scenario_display_string, "\033[39m",
            "\033[0;35m " + "*" * ((100-scenario_display_string.__len__())//2-3)
        )
        print(
            "\033[0;35m ***************************************************************************************************"
        )

        # generate text until the output length (which includes the context length) reaches 50

        print("\033[39m")
        if hi == "Y":
            print("  HINT :" + "\033[1;34m", random.choice(hnt[str(env.state).replace("-","_")]))
            # print("  HINT :" + "\033[1;34m", str(env.state))
            print("\033[39m")
        print(
            "\033[0;35m ***************************************************************************************************"
        )
        print("\033[39m")
        print("  ACTIONS:")

        for i in range(no_of_actions):
            print("\n")
            print("\033[0;35m    ", i, "\033[39m", ":",
                  "\033[1;33m", env.action_spaces[i])
        print("\n")

        print(
            "\033[0;35m ***************************************************************************************************"
        )
        print("\033[39m")
        while True:
            j = 0
            i = input("  Choose an Action: ")
            j = int(i)
            if (not str(i).isdigit()) or int(i) < 0 or int(i) > len(scenario):
                print("\033[0;31m Invalid Input!! Try Again.")
                print("\033[39m")
            else:
                i = int(i)
                print("\033[39m")
                break
        print(
            "\033[0;35m\n ***************************************************************************************************"
        )
        ch = []
        for i in range(len(env.action_spaces)):
            ch.append(str(i)+"." + str(env.action_spaces[i]))
        a = env.action_spaces[j]
        prev = env.state
        asr.append(i)
        sid.append(str(env.state))
        cig.append(ch)
        sc.append(a)
        # ph.append(hnt[str(env.state).replace("-", "_")])
        s, r, done, quest = env.step(j)
        # print("fine\n")
        print("\033[39m")
        print("  Selected Choice : ", a)

        reward += r
        if r <= -1:
            life = life+r
        print("  Point Acquired :", r)
        print("  Total reward : ", reward)
        print("  Lives Left : ", life)
        print("  Percentage completion:", round(env.per_clp * 100, 2), "%\n")
        print("\033[0;36m", u"█" * int(env.per_clp * 90)
              + " " * int((1 - env.per_clp) * 90)
              + "| ",
              round(env.per_clp * 100, 2),
              "%",
              )
        print("\033[39m")
        print(
            "\033[0;35m ***************************************************************************************************"
        )
        print("\033[39m")
        if r >= 0:
            cic.append("Correct")
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            # print("\033[39m")
            print(
                "\033[0;35m ******************************************"+"\033[39m"
                + "\033[1;32m Right Choice!"
                + "\033[39m" + "\033[39m"
                + "\033[0;35m ******************************************"
            )
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            print("\033[39m ")
        else:
            cic.append("Incorrect")
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            # print("\033[39m ")
            print(
                "\033[0;35m **********************************" + "\033[39m"
                + "\033[1;31m WRONG CHOICE! 1 Life Deducted"
                + "\033[39m"
                + "\033[0;35m **********************************"
            )
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            print("\033[39m ")
        if mem == "N":
            time.sleep(2)
            os.system('clear')
        cnt = cnt+1
        cr.append(r)
        gs.append(reward)
        gcp.append(env.per_clp)
        tim.append(time.time())
        # saved.append(l)
        if life <= 0:
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            # print("\033[39m ")
            print(

                "\033[0;35m **************************************" + "\033[39m"
                + "\033[1;31m GAME OVER YOU LOST!!!"
                + "\033[39m"
                + "\033[0;35m **************************************"
            )
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            print("\033[39m ")
            break
        if life > 0 and done:
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            print("\033[39m")
            print(
                "\033[0;35m ***************************************"+"\033[39m"
                + "\033[1;32m GAME OVER YOU WON!!"
                + "\033[39m" + "\033[39m"
                + "\033[0;35m ***************************************"
            )
            print(
                "\033[0;35m ***************************************************************************************************"
            )
            print("\033[39m ")

    fg = time.time()
    q = str(env.quest.partition(".")[0])

    print("\033[39m")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
   #---- environment specifications ----#
    parser.add_argument('--scn',
                        type=str,
                        default='baking a cake',
                        choices=["repairing a flat bicycle tire",
                                 "borrowing a book from the library",
                                 "taking a bath",
                                 "going grocery shopping",
                                 "going on a train",
                                 "planting a tree",
                                 "flying in an airplane",
                                 "baking a cake",
                                 "riding on a bus",
                                 "getting a hair cut", ])

    parser.add_argument('--no_of_actions',
                        type=int,
                        default=5)

    parser.add_argument('--allowed_wrong_actions',
                        type=int,
                        default=5)

    parser.add_argument('--hop',
                        type=int,
                        default=1)

    parser.add_argument('--disclose_state_node',
                        action="store_true",
                        help="shows state nodes/hints",)

    parser.add_argument('--seed',
                        type=int,
                        default=42)

    parser.add_argument(
        "--history",
        action="store_true",
        help="Shows the history of previous steps",
    )

    args = parser.parse_args()

    run_game(args)
