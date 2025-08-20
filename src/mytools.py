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

import base64
from IPython.display import display
from PIL import Image
import sys
import streamlit as st

import os
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, constr
from typing import List, Optional, Literal

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
    file_path: Optional[str] = Field(
        None, description="Optional file path where the PNG image is saved."
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

    # @tool
    # @staticmethod
    # def display_base64_image(base64_string, caption="Image"):
    #     """
    #     Displays a base64-encoded image string as an plot (a pil image object)
    #     """
    #     # Remove the prefix if it exists
    #     if base64_string.startswith("base64_image:"):
    #         base64_string = base64_string[13:]  # Remove "base64_image:" prefix
        
    #     # Decode the base64 string
    #     img_data = base64.b64decode(base64_string)
    #     # print(img_data)
    #     # Create an image from the decoded data
    #     img = Image.open(io.BytesIO(img_data))
    #     # display(Image(img))
    #     try:
    #     # Proper way to check if Streamlit is running
    #         import streamlit as st
    #         # Check if we're in a Streamlit environment
    #         from streamlit.runtime.scriptrunner import get_script_run_ctx
    #         if get_script_run_ctx() is not None:
    #             st.image(img, caption=caption or "Image")
    #             return "Displayed in Streamlit"
    #     except ImportError:
    #         pass
    #     except Exception as e:
    #         print(f"Streamlit display error: {e}")
        
    #     # # Fallback for non-Streamlit environments
    #     # return img  # Return the PIL Image object

    #     if "ipykernel" in sys.modules:  
    #         # Jupyter/Colab
    #         from IPython.display import display
    #         display(img)
    #         return "Displayed in Jupyter"
    #     else:
    #         # Fallback: plain Python -> open system viewer
    #         img.show()
    #         # Save to a temporary file to display
    #     img.save("temp_plot.png")
    #     return "temp_plot.png"

   
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
    
    
    

    # @tool
    # @staticmethod
    # def gen_plot(request: PlotRequest) -> PlotResponse:
    #     """
    #     Generate a plot using Matplotlib. Returns base64-encoded PNG string.
    #     """
    #     x = request.x or [1, 2, 3]
    #     y = request.y or [1, 4, 9]
    #     label = request.label

    #     # Load labels from CSV if provided
    #     if request.path:
    #         try:
    #             df = pd.read_csv(request.path)
    #             label = df.iloc[:, 0].tolist()
    #         except Exception as e:
    #             label = None
    #             print(f"Warning: Could not read labels from {request.path} - {e}")

    #     plt.figure()

    #     if request.plot_type == "line":
    #         plt.plot(x, y)
    #     elif request.plot_type == "bar":
    #         plt.bar(x, y)
    #     elif request.plot_type == "scatter":
    #         plt.scatter(x, y, c=label) if label is not None else plt.scatter(x, y)
    #     else:
    #         raise ValueError(f"Unsupported plot_type: {request.plot_type}")

    #     plt.title(request.title)
    #     plt.xlabel("X")
    #     plt.ylabel("Y")

    #     buf = io.BytesIO()
    #     plt.savefig(buf, format="png")
    #     plt.close()
    #     buf.seek(0)

    #     img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    #     return PlotResponse(base64_string=img_base64)


    @tool
    @staticmethod
    def gen_plot(request: PlotRequest) -> PlotResponse:
        """
        Generate a plot using Matplotlib. Saves the plot to a directory and returns the filepath.
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
        plt.show()
        # Create directory if it doesn't exist
        plot_dir = "generated_plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"plot_{timestamp}_{unique_id}.png"
        filepath = os.path.join(plot_dir, filename)
        
        # Save the plot to file
        plt.savefig(filepath, format="png", dpi=100, bbox_inches='tight')
        plt.close()
        
        # Also create base64 for backward compatibility
        with open(filepath, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        
        print(f"Plot saved to: {filepath}")
        
        # Return both filepath and base64
        # return PlotResponse(filepath=filepath,)
        return PlotResponse(base64_string=img_base64, filepath=filepath)


    # @tool ## Works
    # @staticmethod
    # def display_base64_image(input_source, caption="Image"):
    #     """
    #     Displays an image from either a file path, base64-encoded string, or reads from directory.
        
    #     Args:
    #         input_source: Can be a filepath, base64 string, or 'latest' to get the most recent plot
    #         caption: Caption for the image
    #     """
    #     img = None
    #     source_file = None
        
    #     # Check if we should get the latest file from directory
    #     if input_source == "latest" or input_source is None:
    #         plot_dir = "generated_plots"
    #         if os.path.exists(plot_dir):
    #             files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    #             if files:
    #                 # Get the most recent file
    #                 files_with_path = [os.path.join(plot_dir, f) for f in files]
    #                 source_file = max(files_with_path, key=os.path.getctime)
    #                 img = Image.open(source_file)
    #                 print(f"Loading latest plot: {source_file}")
        
    #     # Check if input is a file path
    #     elif isinstance(input_source, str) and (os.path.exists(input_source) or 
    #                                         os.path.exists(os.path.join("generated_plots", input_source))):
    #         # Try direct path first
    #         if os.path.exists(input_source):
    #             source_file = input_source
    #         else:
    #             # Try in generated_plots directory
    #             source_file = os.path.join("generated_plots", input_source)
            
    #         img = Image.open(source_file)
    #         print(f"Loading plot from: {source_file}")
        
    #     # Handle base64 string
    #     elif isinstance(input_source, str):
    #         base64_string = input_source
            
    #         # Remove the prefix if it exists
    #         if base64_string.startswith("base64_image:"):
    #             base64_string = base64_string[13:]
            
    #         try:
    #             # Decode the base64 string
    #             img_data = base64.b64decode(base64_string)
    #             img = Image.open(io.BytesIO(img_data))
    #             print("Loaded image from base64 string")
    #         except Exception as e:
    #             print(f"Error decoding base64: {e}")
    #             return f"Error: Could not decode base64 string - {e}"
        
    #     if img is None:
    #         return "Error: Could not load image from provided source"
        
    #     # Try to display in different environments
    #     try:
    #         # Check if Streamlit is available and running
    #         import streamlit as st
    #         from streamlit.runtime.scriptrunner import get_script_run_ctx
            
    #         if get_script_run_ctx() is not None:
    #             # We're in a Streamlit environment
    #             if source_file:
    #                 # Display from file (more efficient for Streamlit)
    #                 st.image(source_file, caption=caption)
    #             else:
    #                 # Display from PIL Image
    #                 st.image(img, caption=caption)
    #             return f"Displayed in Streamlit: {caption}"
    #     except ImportError:
    #         pass
    #     except Exception as e:
    #         print(f"Streamlit display error: {e}")
        
    #     # Check for Jupyter/IPython environment
    #     if "ipykernel" in sys.modules:
    #         try:
    #             from IPython.display import display
    #             display(img)
    #             return f"Displayed in Jupyter: {caption}"
    #         except Exception as e:
    #             print(f"Jupyter display error: {e}")
        
    #     # Fallback: save to temp file and return path
    #     temp_path = "temp_plot.png"
    #     img.save(temp_path)
    #     print(f"Image saved to: {temp_path}")
        
    #     # Try to open with system viewer
    #     try:
    #         img.show()
    #         return f"Opened in system viewer and saved to: {temp_path}"
    #     except:
    #         return f"Image saved to: {temp_path}"

    @tool
    @staticmethod
    def display_base64_image(file_path: str | None = None,
        base64_string: str | None = None,
        caption: str = "Image"):
        """Displays an image from either a file path, base64-encoded string, or reads from directory."""

        result = {"status": "ok", "caption": caption}
        # Load from file path
        if file_path and os.path.exists(file_path):
            st.image(file_path, caption=caption)
            return {"status": "ok", "caption": caption, "filepath": file_path}
            # with open(file_path, "rb") as f:
            #     b64 = base64.b64encode(f.read()).decode("utf-8")
            # result["filepath"] = file_path
            # result["data_url"] = f"data:image/png;base64,{b64}"
            # return result

        # Load from base64
        if base64_string:
            if base64_string.startswith("base64_image:"):
                base64_string = base64_string[13:]

            if base64_string.startswith("data:image/png;base64,"):
                base64_string = base64_string.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=caption)
            return {"status": "ok", "caption": caption}
            # # Ensure data URL form
            # prefix = "data:image/png;base64,"
            # result["data_url"] = base64_string if base64_string.startswith(prefix) else prefix + base64_string
            # return result
        return {"status": "error", "message": "No image found"}

    # import re
    # import os
    # import io
    # import base64
    # import sys
    # from PIL import Image

    # @tool
    # @staticmethod
    # def display_base64_image(
    #     file_path: str | None = None,
    #     base64_string: str | None = None,
    #     caption: str = "Image"
    # ):
    #     """
    #     Displays an image either from:
    #     - saved file path,
    #     - base64 string, or
    #     - markdown attachment syntax: ![caption](attachment://file.png)

    #     Works in Streamlit, Jupyter, or plain Python.
    #     """
    #     img = None

    #     # --- Parse Markdown attachment ---
    #     if file_path and file_path.startswith("!["):
    #         # Example: "![Bar Plot](attachment://plot.png)"
    #         match = re.match(r"!\[(.*?)\]\(attachment://(.*?)\)", file_path)
    #         if match:
    #             caption = match.group(1) or caption
    #             file_path = match.group(2)

    #     # --- Load from file path ---
    #     if file_path and os.path.exists(file_path):
    #         img = Image.open(file_path)

    #     #--- Load from base64 ---
    #     elif base64_string:
    #         if base64_string.startswith("base64_image:"):
    #             base64_string = base64_string[13:]
    #         img_data = base64.b64decode(base64_string)
    #         img = Image.open(io.BytesIO(img_data))

    #     if img is None:
    #         return "No image found"

    #     # --- Try Streamlit ---
    #     try:
    #         import streamlit as st
    #         from streamlit.runtime.scriptrunner import get_script_run_ctx
    #         if get_script_run_ctx() is not None:
    #             st.image(img, caption=caption or "Image")
    #             return "Displayed in Streamlit"
    #     except ImportError:
    #         pass
    #     except Exception as e:
    #         print(f"Streamlit display error: {e}")

    #     # --- Jupyter Notebook ---
    #     if "ipykernel" in sys.modules:
    #         from IPython.display import display
    #         display(img)
    #         return "Displayed in Jupyter"

    #     # --- Fallback: System viewer ---
    #     img.show()
    #     return file_path or "Image shown"
