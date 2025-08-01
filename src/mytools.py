from langchain.tools import tool
from scipy.stats import shapiro, levene, f
from scipy.linalg import eigh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import requests
import io
import base64




class MyTools:
####################################################### TOOLS ############################################################

    @tool
    @staticmethod
    def calculate_PC(path: str):
        '''Calculate principal components'''
        data = pd.read_csv(path)
        mu = data.to_numpy().mean(axis=0)
        std = data.to_numpy().std(axis=0)
        standardized_data = (data.to_numpy() - mu) / std
        cov_matrix = np.cov(standardized_data, rowvar=False)
        eigenvalue, eigenvector = eigh(cov_matrix)
        eigenvalue = eigenvalue[::-1]
        eigenvector = eigenvector[:, ::-1]
        transformed_data = standardized_data @ np.vstack((eigenvector[:, 0], eigenvector[:, 1])).T
        return [transformed_data[:, 0].tolist(), transformed_data[:, 1].tolist()]

    @tool
    @staticmethod
    def levene_test(path: str):
        '''Calculate levene's test'''
        data = pd.read_csv(path)
        data = np.array(data).T
        N = data.shape[0] * data.shape[1]
        k = data.shape[1]
        Ni = data.shape[0]
        Zij = [np.abs(data.T[i] - data.T[i].mean()) for i in range(k)]
        Zi = [np.sum(Zij[i]) / Ni for i in range(k)]
        Z = np.sum([np.sum(Zij[i]) for i in range(k)]) / N
        W = ((N - k) / (k - 1)) * (np.sum(Ni * (np.array(Zi) - Z) ** 2) / np.sum([np.sum((Zij[i] - Zi[i]) ** 2) for i in range(k)]))
        pval = 1 - f.cdf(W, k - 1, N - k)
        return W, pval

    @tool
    @staticmethod
    def anova_one_way(path: str):
        '''Calculate one-way-anova'''
        dd = pd.read_csv(path)               
        k = dd.shape[1]
        n = dd.shape[0]
        N = k * n
        X = np.array([dd[col].mean() for col in dd.columns]).mean()
        Xj = np.array([dd[col].mean() for col in dd.columns])
        Xij = np.array([dd[col] for col in dd.columns])
        SSR = n * np.sum((Xj - X) ** 2)
        SSE = np.sum([np.sum((Xij[i] - Xj[i]) ** 2) for i in range(len(Xj))])
        MS = SSR / (k - 1)
        MSE = SSE / (N - k)
        F_stat = MS / MSE
        pval = 1 - f.cdf(F_stat, k - 1, N - k)
        return F_stat, pval

    @tool
    @staticmethod
    def covariance_tool(path: str):
        '''Calculate covariance'''
        df = pd.read_csv(path)
        return df.cov()

    @tool
    @staticmethod
    def correlation_tool(path: str):
        '''Calculate correlation'''
        df = pd.read_csv(path)
        return df.corr()
        
    @tool
    @staticmethod
    def normality_tool(path: str):
        '''Calculate if the data is normal'''
        df = pd.read_csv(path)
        for col in df.columns:
            tstat, pval = shapiro(df[col])
            if pval < 0.05:
                return f'Column {col} is not normally distributed, and not recommended to include in ANOVA analysis'
        return 'Data is normal'

    # @tool
    # @staticmethod
    # def gen_plot(plot_type: str = "line", title: str = "Plot", x: list = [1, 2, 3], y: list = [1, 4, 9], label: list = [1, 2, 3], path: str | None=None) -> str:
    #     """Generate a plot using Matplotlib. Returns base64-encoded PNG string.
        
    #     Args:
    #         plot_type: 'line', 'bar', or 'scatter'
    #         title: title of the plot
    #         x: list of x values
    #         y: list of y values
    #     """
    #     label = [1, 2, 3] or pd.read_csv(path)
    #     plt.figure()
    #     if plot_type == "line":
    #         plt.plot(x, y)
    #     elif plot_type == "bar":
    #         plt.bar(x, y)
    #     elif plot_type == "scatter":
    #         plt.scatter(x, y, c=label)
    #     else:
    #         return f"Unsupported plot_type: {plot_type}"

    #     plt.title(title)
    #     plt.xlabel("X")
    #     plt.ylabel("Y")

    #     buf = io.BytesIO()
    #     # plt.savefig(buf, format='png')
    #     # plt.close()
    #     buf.seek(0)
    #     # print('image has been created')
    #     img_bytes = buf.read()
    #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    #     return 'image created'
    @tool
    @staticmethod
    def gen_plot(plot_type: str = "line", title: str = "Plot", x: list | None = None, y: list | None = None, label: list | None = None,path: str | None = None) -> str:
        """
        Generate a plot using Matplotlib. Returns base64-encoded PNG string.
        
        Args:
            plot_type: 'line', 'bar', or 'scatter'
            title: title of the plot
            x: list of x values
            y: list of y values
            label: list of labels (only used for scatter)
            path: optional path to CSV for labels
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import io
        import base64

        # Defaults
        if x is None:
            x = [1, 2, 3]
        if y is None:
            y = [1, 4, 9]

        # Load labels from CSV if provided
        if path:
            try:
                df = pd.read_csv(path)
                label = df.iloc[:, 0].tolist()
            except Exception as e:
                label = None
                print(f"Warning: Could not read labels from {path} - {e}")

        plt.figure()

        if plot_type == "line":
            plt.plot(x, y)
        elif plot_type == "bar":
            plt.bar(x, y)
        elif plot_type == "scatter":
            if label is not None:
                plt.scatter(x, y, c=label)
            else:
                plt.scatter(x, y)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")

        buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return 'image created'
