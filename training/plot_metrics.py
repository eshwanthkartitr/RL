import json
import matplotlib.pyplot as plt

def plot_metrics():
    try:
        with open("outputs/eval_results.json", "r") as f:
            data = json.load(f)
            
        labels = ["Naive Baseline", "Rule Baseline"]
        values = [data["naive_avg"], data["rule_avg"]]
        
        plt.bar(labels, values, color=["red", "blue"])
        plt.ylabel("Average Reward")
        plt.title("Evaluation: Baselines")
        plt.savefig("outputs/eval_chart.png")
        print("Chart saved to outputs/eval_chart.png")
        
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    plot_metrics()
