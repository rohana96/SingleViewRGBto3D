"""
Usage:
    python main.py
"""

import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='all', type=str)
    args = parser.parse_args()

    if args.command in ['all', 'q1.1']:

        # 1.1
        os.system("python fit_data.py --type vox")

    if args.command in ['all', 'q1.2']:
        # 1.2
        os.system("python fit_data.py --type point")

    if args.command in ['all', 'q1.3']:
        # 1.3
        os.system("python fit_data.py --type mesh")

    if args.command in ['all', 'q2.1']:
        # 2.1
        os.system("python train_model.py --type vox")
        os.system("python eval_model.py --type vox")

    if args.command in ['all', 'q2.2']:
        # 2.2
        os.system("python train_model.py --type point")
        os.system("python eval_model.py --type point")

    if args.command in ['all', 'q2.3']:
        # 2.3
        os.system("python train_model.py --type mesh")
        os.system("python eval_model.py --type mesh")

    if args.command in ['all', 'q2.4']:
        # 2.4
        os.system("python eval_model.py --type vox")
        os.system("python eval_model.py --type point")
        os.system("python eval_model.py --type mesh")

    if args.command in ['all', 'q2.5']:
        # 2.5
        os.system("python train_model.py --type mesh --w_smooth 0.1")
        os.system("python train_model.py --type point --w_smooth 0.3")
        os.system("python train_model.py --type mesh --w_smooth 0.5")

        os.system("python train_model.py --type mesh --w_smooth 0.1")
        os.system("python train_model.py --type mesh --w_smooth 0.3")
        os.system("python train_model.py --type mesh --w_smooth 0.5")


    if args.command in ['all', 'q2.6']:
        # 2.6
        os.system("python latent_space_edit.py --type point")

    if args.command in ['all', 'q3.1']:
        # 3.1
        os.system("python train_implicit.py --type vox")
        os.system("python eval_implicit.py --type vox")


if __name__ == "__main__":
    main()

