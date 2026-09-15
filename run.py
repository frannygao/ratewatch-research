import argparse

from ratewatch.audit import issue_forecast, score_log
from ratewatch.forecast import evaluate, forecast_next, save_results
from ratewatch.sources import load_data


def main():
    parser = argparse.ArgumentParser(description='Bank of Canada cut, hold and hike probabilities.')
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--saved', dest='offline', action='store_true',
                         help='Rerun the saved data at its saved date, without downloading.')
    options.add_argument('--record', dest='issue', action='store_true',
                         help='Save a dated forecast for later evaluation. Requires fresh downloads.')
    options.add_argument('--offline', dest='offline', action='store_true', help=argparse.SUPPRESS)
    options.add_argument('--issue', dest='issue', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    data = load_data(offline=args.offline)
    evaluation = evaluate(data)
    forecast = forecast_next(data)
    if args.issue:
        forecast = issue_forecast(data, evaluation, forecast)
    prospective = score_log(data['folder'] / 'audit', data['meetings'])
    report = save_results(evaluation, forecast, prospective)
    print(f"Evaluated {evaluation['n']} meetings. Model accuracy: {evaluation['scores']['model']['accuracy']:.1%}.")
    if 'probabilities' in forecast:
        print(f"{forecast['meeting_date']} ({forecast['status']}): " + ', '.join(
            f'{label} {value:.1%}' for label, value in forecast['probabilities'].items()))
    print(f'Report: {report}')


if __name__ == '__main__':
    main()
