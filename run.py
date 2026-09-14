import argparse

from ratewatch.audit import issue_forecast, score_log
from ratewatch.forecast import evaluate, forecast_next, save_results
from ratewatch.sources import load_data


def main():
    parser = argparse.ArgumentParser(description='Bank of Canada cut, hold and hike probabilities.')
    parser.add_argument('--offline', action='store_true', help='Use the saved data and its original cutoff.')
    parser.add_argument('--issue', action='store_true', help='Record a prospective forecast after fresh downloads.')
    args = parser.parse_args()
    if args.offline and args.issue:
        parser.error('Issuing requires fresh downloads; --offline cannot be combined with --issue.')

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
