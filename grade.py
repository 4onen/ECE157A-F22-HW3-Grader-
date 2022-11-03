import json
import csv
from math import floor
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, Union
import numpy as np

expected_shape: Final[Tuple[int, int]] = (1200, 1)

defaultResult: Final[Dict[str, Any]] = \
    {'output': 'Encountered unexpected error trying to grade assignment. Contact course staff! FILE_ERROR_DAT',
     'score': 0.0,
     'visibility': 'visible',
     'stdout_visibility': 'hidden',
     'tests': [],
     'leaderboard': []
     }


def load_outcomes(filename: Union[str, Path]) -> Tuple[List[str], np.ndarray]:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        columnNames = next(reader)
        if columnNames != ['failureType']:
            print(filename, ': Nonstandard column names: ', columnNames)
        rows = np.array(list(reader), dtype=str)
        if rows.shape != expected_shape:
            print(filename, ': Nonstandard data shape: ', rows.shape)
    return columnNames, rows


def mktest(name: str, maxscore: float, passes: bool, output: str = None):
    score = maxscore if passes else 0.0
    test = {'name': name, 'score': score, 'max_score': maxscore}
    if output is not None:
        test['output'] = output
    return test


def mkleaderboardentry(name: str, value: float, asc=False):
    return {'name': name, 'value': floor(value*100)}


if __name__ == '__main__':
    result = defaultResult
    try:
        _, testy = load_outcomes('testy.csv')
        assert(testy.shape == expected_shape)

        from sys import argv
        filename: str = argv[1] if len(argv) > 1 else 'submission/scores.csv'
        try:
            columnNames, predy = load_outcomes(filename)
        except StopIteration:
            result['output'] = "Tried to open {} but couldn't retrieve any rows! FILE_ERROR_DAT."\
                .format(filename)
            raise StopIteration

        if predy.shape != expected_shape:
            result['output'] = "The shape of your data in {} didn't match the assignment's expectations!\
 Assignment expected: {}, Your data: {}. Your column names were: {}. FILE_ERROR_DAT."\
                    .format(filename, expected_shape, predy.shape, columnNames)
        assert(predy.shape == expected_shape)

        from sklearn.metrics import accuracy_score
        answers = np.unique(testy)
        accuracy = accuracy_score(testy, predy)
        scores = {answer: accuracy_score(testy[testy == answer],
                                         predy[testy == answer])
                  for answer in answers}
        # code = bool(list(Path('submission').glob('*.py'))
        #             or list(Path('submission').glob('*.ipynb')))

        result['tests'] = [
            mktest(f'Overall Accuracy Meets Threshold',
                   1.0, accuracy >= 0.9),
            # mktest(f'Code submitted',
            #        0.5, code,
            #        output=None if code else "No code submitted! We expected to find a .py or .ipynb file in your submission folder."),
        ]

        result['leaderboard'] = [
            mkleaderboardentry('Total Accuracy', accuracy),
        ] + [
            mkleaderboardentry(f'"{answer}" Accuracy', scores[answer])
            for answer in answers
        ]

        del result['output']
        del result['score']
    finally:
        with open('results/results.json', 'w') as outfile:
            json.dump(result, outfile)
