import numpy as np

class Regressor:
    def __init__(self, X: np.ndarray, y: np.ndarray, functions: list["function"], precise: float = 3, beta: float = 0.05) -> None:
        self.X = X
        self.y = y
        self.functions = functions
        self.beta = beta
        self.psi_matrix = np.array([[f(val) for f in self.functions] for val in self.X])
        self.F_inverted = self.findFInvertedMatrix()
        self.beta_estimations = self.findRegressionCoefficients()
        self.precise = precise

    def findFInvertedMatrix(self) -> np.ndarray:
        F_matrix = self.psi_matrix.T @ self.psi_matrix
        F_inverted_matrix = np.linalg.inv(F_matrix)

        return F_inverted_matrix

    def findRegressionCoefficients(self) -> np.ndarray:
        F_matrix = self.psi_matrix.T @ self.psi_matrix
        F_inverted_matrix = np.linalg.inv(F_matrix)
        beta_estimations = F_inverted_matrix @ self.psi_matrix.T @ self.y

        return beta_estimations
    
    def findErrors(self) -> np.ndarray:
        self.beta_estimations = self.findRegressionCoefficients()
        y_hat = self.psi_matrix @ self.beta_estimations
        return self.y - y_hat
    
    def findRSS(self) -> float:
        errors = self.findErrors()
        return errors.T @ errors
    
    def findTSS(self) -> float:
        y_avg = np.average(self.y)
        return np.sum((y_avg - self.y) ** 2)
    
    def findRSquare(self) -> float:
        TSS = self.findTSS()
        RSS = self.findRSS()
        return np.round((TSS - RSS) / TSS, self.precise)
    
    def checkCoefficientSignificance(self, ind: int) -> float:
        RSS = self.findRSS()
        F_inverted_i_i = self.F_inverted[ind, ind]
        beta_estimated = self.beta_estimations[ind]
        n = self.X.shape[0]
        p = len(self.functions)
        delta_estimation = abs(beta_estimated / np.sqrt(RSS * F_inverted_i_i / (n - p)))
        p_value = t.cdf(-delta_estimation, n - p) + t.sf(delta_estimation, n - p)
        return np.round(p_value, self.precise)
    
    def findConfidenceInterval(self, point: np.ndarray) -> tuple[float]:
        prediction = self.getPrediction(point)
        n = self.X.shape[0]
        p = len(self.functions)
        RSS = self.findRSS()
        func_vals = np.array([func(point) for func in self.functions])
        val = np.sqrt(RSS * (1 + func_vals @ self.F_inverted @ func_vals.T) / (n - p))
        left_bound = (1 - self.beta) / 2
        right_bound = (1 + self.beta) / 2
        left_percentile = t.ppf(left_bound, n - p)
        right_percentile = t.ppf(right_bound, n - p)

        left_value = prediction - val * right_percentile
        right_value = prediction - val * left_percentile

        return (left_value, right_value)

    def getPrediction(self, point: np.ndarray) -> float:
        func_vals = np.array([func(point) for func in self.functions])
        prediction = self.beta_estimations.T @ func_vals
        return prediction
    
    def getPredictions(self, points: np.ndarray) -> np.ndarray:
        predictions = [self.getPrediction(point) for point in points]
        return predictions
    
    def findIndependance(self) -> float:
        errors = self.findErrors()
        n = errors.shape[0]
        inv_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (errors[j] < errors[i]):
                    inv_count += 1
        
        middle_inv_count = n * (n - 1) // 4

        delta = abs((inv_count - middle_inv_count) / np.sqrt(n ** 3 / 36))
        p_value = norm.cdf(-delta) + norm.sf(delta)
        return p_value
    
    def printRegr(self) -> None:
        answ = f"{np.round(self.beta_estimations[0], self.precise)}"
        for i in range(1, self.beta_estimations.shape[0]):
            coef = self.beta_estimations[i]
            answ += f" + {str(np.round(coef, self.precise))}*x_{i}"
        print("y =", answ)