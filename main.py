import argparse
import pandas as pd
import sys
from eda import run_eda
from preprocess import preprocess_data
from train_models import train_and_return_models
from evaluate import evaluate_models, print_model_comparison
from utils import save_model, infer_task_type


def main():
    parser = argparse.ArgumentParser(description='AutoML CLI Tool')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], help='Task type (auto-inferred if not provided)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--output', type=str, default='best_model.pkl', help='Output path for saved model')
    args = parser.parse_args()

    # Load dataset
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    print("\n--- Running EDA ---")
    run_eda(df, args.target)

    print("\n--- Preprocessing Data ---")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, args.target, args.test_size)

    # Infer task type if not provided
    task_type = args.task or infer_task_type(df, args.target)
    print(f"\nTask type: {task_type}")

    print("\n--- Training Models ---")
    models = train_and_return_models(X_train, y_train, task_type)

    print("\n--- Evaluating Models ---")
    results = evaluate_models(models, X_test, y_test, task_type)
    print_model_comparison(results, task_type)

    # Ask user to select model
    print("\nSelect a model to save:")
    for i, model_name in enumerate(results.keys()):
        print(f"{i+1}. {model_name}")
    selected = input("Enter the number of the model to save: ")
    try:
        selected_idx = int(selected) - 1
        selected_model_name = list(results.keys())[selected_idx]
    except Exception:
        print("Invalid selection. Exiting.")
        sys.exit(1)
    selected_model = models[selected_model_name]
    save_model(selected_model, args.output)
    print(f"Model '{selected_model_name}' saved to {args.output}")

if __name__ == '__main__':
    main() 