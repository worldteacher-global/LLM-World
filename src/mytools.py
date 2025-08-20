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
from pydantic import BaseModel
from typing import List
import base64
from IPython.display import display
from PIL import Image
import sys

from pydantic import BaseModel, Field, constr
from typing import List, Optional, Literal
import matplotlib.pyplot as plt
import pandas as pd
import io, base64
class PlotRequest(BaseModel):
    plot_type: Literal["line", "bar", "scatter"] = Field(default="line")
    title: str = Field(default="Plot")
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    label: Optional[List[float]] = None
    path: Optional[str] = Field(
        default=None,
        description="Optional CSV path; if provided, first column used as labels"
    )


class PlotResponse(BaseModel):
    base64_string: constr(min_length=10) = Field(
        ..., description="Base64-encoded PNG image string."
    )



class DataFramePayload(BaseModel):
    columns: List[str]
    data: List[List[float]]

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
        payload = {
            "PC1":transformed_data[:,0].tolist(),
            "PC2":transformed_data[:,1].tolist()}
        return payload

    @tool
    @staticmethod
    def levene_test(path: str):
        '''Calculate levene's test for equality of variances'''
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
        payload = {
            "test":"levene's test",
            "W":float(W),
            "Pval":pval}
        return payload

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
        payload = {
            "test":"one_way_anova",
            "F":float(F_stat),
            "P_val":float(pval)}
        return payload

    @tool
    @staticmethod
    def covariance_tool(path: str):
        '''Calculate covariance'''
        df = pd.read_csv(path)
        cov = df.cov()
        return DataFramePayload(
            columns=df.columns.tolist(),
            data=cov.values.tolist())

    @tool
    @staticmethod
    def correlation_tool(path: str):
        '''Calculate correlation'''
        df = pd.read_csv(path)
        corr = df.corr()
        return DataFramePayload(
            columns=df.columns.tolist(),
            data=corr.values.tolist())
        
    @tool
    @staticmethod
    def normality_tool(path: str):
        '''Calculate if the data is normal'''
        df = pd.read_csv(path)
        for col in df.columns:
            tstat, pval = shapiro(df[col])
            if pval < 0.05:
                return {'result':f'Column {col} is not normally distributed, and not recommended to include in ANOVA analysis'}
        return {'result':'Data is normal'}

    @tool
    @staticmethod
    def display_base64_image(base64_string):
        """
        Displays a base64-encoded image string
        """
        # Remove the prefix if it exists
        if base64_string.startswith("base64_image:"):
            base64_string = base64_string[13:]  # Remove "base64_image:" prefix
        
        # Decode the base64 string
        img_data = base64.b64decode(base64_string)
        print(img_data)
        # Create an image from the decoded data
        img = Image.open(io.BytesIO(img_data))
        # display(Image(img))
        try:
        # Proper way to check if Streamlit is running
            import streamlit as st
            # Check if we're in a Streamlit environment
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                st.image(img, caption=caption or "Image")
                return "Displayed in Streamlit"
        except ImportError:
            pass
        except Exception as e:
            print(f"Streamlit display error: {e}")
        
        # Fallback for non-Streamlit environments
        return img  # Return the PIL Image object

    # print("Function created successfully!")
    # print("\nKey issues in your original code:")
    # print("1. 'caption' parameter was missing from function signature")
    # print("2. 'st._is_running_with_streamlit' is not a reliable check")
    # print("3. 'print(img_data)' was printing raw binary data")
    # print("4. Missing proper error handling")

        # if "ipykernel" in sys.modules:  
        #     # Jupyter/Colab
        #     from IPython.display import display
        #     display(img)
        #     return "Displayed in Jupyter"
        # else:
        #     # Fallback: plain Python -> open system viewer
        #     img.show()
        #     # Save to a temporary file to display
        # img.save("temp_plot.png")
        # return "temp_plot.png"

   
    # @tool
    # @staticmethod
    # def gen_plot(plot_type: str = "line", title: str = "Plot", x: list | None = None, y: list | None = None, label: list | None = None,path: str | None = None) -> str:
    #     """
    #     Generate a plot using Matplotlib. Returns base64-encoded PNG string.
        
    #     Args:
    #         plot_type: 'line', 'bar', or 'scatter'
    #         title: title of the plot
    #         x: list of x values
    #         y: list of y values
    #         label: list of labels (only used for scatter)
    #         path: optional path to CSV for labels
    #     """
    #     import matplotlib.pyplot as plt
    #     import pandas as pd
    #     import io
    #     import base64

    #     # Defaults
    #     if x is None:
    #         x = [1, 2, 3]
    #     if y is None:
    #         y = [1, 4, 9]

    #     # Load labels from CSV if provided
    #     if path:
    #         try:
    #             df = pd.read_csv(path)
    #             label = df.iloc[:, 0].tolist()
    #         except Exception as e:
    #             label = None
    #             print(f"Warning: Could not read labels from {path} - {e}")

    #     plt.figure()

    #     if plot_type == "line":
    #         plt.plot(x, y)
    #     elif plot_type == "bar":
    #         plt.bar(x, y)
    #     elif plot_type == "scatter":
    #         if label is not None:
    #             plt.scatter(x, y, c=label)
    #         else:
    #             plt.scatter(x, y)
    #     else:
    #         raise ValueError(f"Unsupported plot_type: {plot_type}")

    #     plt.title(title)
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.show()

    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     plt.close()
    #     buf.seek(0)

    #     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    #     payload = {'base64_string':img_base64}
    #     return payload
    
    
    

    @tool
    @staticmethod
    def gen_plot(request: PlotRequest) -> PlotResponse:
        """
        Generate a plot using Matplotlib. Returns base64-encoded PNG string.
        """
        x = request.x or [1, 2, 3]
        y = request.y or [1, 4, 9]
        label = request.label

        # Load labels from CSV if provided
        if request.path:
            try:
                df = pd.read_csv(request.path)
                label = df.iloc[:, 0].tolist()
            except Exception as e:
                label = None
                print(f"Warning: Could not read labels from {request.path} - {e}")

        plt.figure()

        if request.plot_type == "line":
            plt.plot(x, y)
        elif request.plot_type == "bar":
            plt.bar(x, y)
        elif request.plot_type == "scatter":
            plt.scatter(x, y, c=label) if label is not None else plt.scatter(x, y)
        else:
            raise ValueError(f"Unsupported plot_type: {request.plot_type}")

        plt.title(request.title)
        plt.xlabel("X")
        plt.ylabel("Y")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return PlotResponse(base64_string=img_base64)