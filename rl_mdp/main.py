from util import create_mdp
from util import create_policy_1
from util import create_policy_2
from model_free_prediction.monte_carlo_evaluator import MCEvaluator
from model_free_prediction.td_evaluator import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    
    n = 10000
    MDP = create_mdp()
    policy1 = create_policy_1()
    policy2 = create_policy_2()
    evaluator = MCEvaluator(MDP)
    print(f"MCevaluator policy 1: {evaluator.evaluate(policy1, n)}")
    print(f"MCevaluator policy 2: {evaluator.evaluate(policy2, n)}")
    
    evaluator = TDEvaluator(MDP, alpha=0.1)
    print(f"TDevaluator policy 1: {evaluator.evaluate(policy1, n)}")
    print(f"TDevaluator policy 2: {evaluator.evaluate(policy2, n)}")
    
    evaluator = TDLambdaEvaluator(MDP, alpha=0.1, lambd=0.5)
    print(f"TDlambdaevaluator policy 1: {evaluator.evaluate(policy1, n)}")
    print(f"TDlambdaevaluator policy 2: {evaluator.evaluate(policy2, n)}")
    


if __name__ == "__main__":
    main()
