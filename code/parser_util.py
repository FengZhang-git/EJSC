# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFile',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--vocabFile',
                        type=str,
                        help='path to intent vocab')

    parser.add_argument('--fileVocab',
                        type=str,
                        help='path to pretrained model vocab')
        
    parser.add_argument('--fileModelConfig',
                        type=str,
                        help='path to pretrained model config')


    parser.add_argument('--fileModel',
                        type=str,
                        help='path to pretrained model')

    parser.add_argument('--fileModelSave',
                        type=str,
                        help='path to save model')


    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('--num_episode_per_epoch',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('--num_episode_val',
                        type=int,
                        help='number of episodes for val, default=60',
                        default=60)

    parser.add_argument('--num_episode_test',
                        type=int,
                        help='number of episodes for testing, default=60',
                        default=5)

    parser.add_argument('--numFreeze',
                        type=int,
                        help='number of freezed layers in pretrained model, default=12',
                        default=12)


    parser.add_argument('--numDevice',
                        type=int,
                        help='id of gpu ',
                        default=0)

    parser.add_argument('--warmup_steps',
                        type=int,
                        help='num of warmup_steps',
                        default=100)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='ratio of decay',
                        default=0.2)

    parser.add_argument('--dropout_rate',
                        type=float,
                        help='ratio of dropout',
                        default=0.1)
    parser.add_argument('--temperature',
                        type=float,
                        help='temperature of scl',
                        default=0.1)
    parser.add_argument('--lamda1',
                        type=float,
                        help='1-ratio of slot in pn',
                        default=0.1)
    parser.add_argument('--lamda2',
                        type=float,
                        help='2-ratio of intent in scl',
                        default=0.1)
    parser.add_argument('--lamda3',
                        type=float,
                        help='3-ratio of slot in scl',
                        default=0.1)

    return parser
