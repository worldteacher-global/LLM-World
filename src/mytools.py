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
        payload = {
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
        payload = {"test":"one_way_anova",
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
    # def display_base64_image(base64_string):
    #     """
    #     Displays a base64-encoded image string
    #     """
    #     # Remove the prefix if it exists
    #     if base64_string.startswith("base64_image:"):
    #         base64_string = base64_string[13:]  # Remove "base64_image:" prefix
        
    #     # Decode the base64 string
    #     img_data = base64.b64decode(base64_string)
        
    #     # Create an image from the decoded data
    #     img = Image.open(io.BytesIO(img_data))
        
    #     # Save to a temporary file to display
    #     img.save("temp_plot.png")
    #     return img

    # @tool
    # @staticmethod
    # def gen_plot(plot_type: str = "line", 
    #             title: str = "Plot", 
    #             x: [list] = None, 
    #             y: [list] = None, 
    #             label: [list] = None, 
    #             path: [str] = None) -> str:
    #     """
    #     Generates a plot using Matplotlib and returns a prefixed base64-encoded PNG string.
    #     Args:
    #         plot_type: 'line', 'bar', or 'scatter'
    #         title: title of the plot
    #         x: list of x values
    #         y: list of y values
    #         label: list of labels (only used for scatter with colors)
    #         path: optional path to a CSV file for data
    #     """
    #     try:
    #         # Import inside function to avoid issues
    #         import matplotlib
    #         matplotlib.use('Agg')  # Use non-interactive backend - CRITICAL FIX
    #         import matplotlib.pyplot as plt
    #         import pandas as pd
    #         import numpy as np
    #         import io
    #         import base64
            
    #         # Clear any existing plots - IMPORTANT
    #         plt.clf()
    #         plt.close('all')
            
    #         # Create new figure
    #         fig = plt.figure(figsize=(10, 6))
            
    #         # If path is provided, try to load data from it
    #         if path:
    #             try:
    #                 df = pd.read_csv(path)
    #                 if df.empty:
    #                     return "Error: CSV file is empty"
    #                 x = df.iloc[:, 0].tolist()
    #                 y = df.iloc[:, 1].tolist() if df.shape[1] > 1 else [0] * len(x)
    #                 if df.shape[1] > 2:
    #                     label = df.iloc[:, 2].tolist()
    #             except FileNotFoundError:
    #                 return f"Error: File not found - {path}"
    #             except Exception as e:
    #                 return f"Error: Could not read data from {path} - {str(e)}"
            
    #         # Use default data if none is provided
    #         if x is None or len(x) == 0: 
    #             x = [1, 2, 3]
    #         if y is None or len(y) == 0: 
    #             y = [1, 4, 9]
                
    #         # Ensure x and y have same length
    #         min_len = min(len(x), len(y))
    #         x = x[:min_len]
    #         y = y[:min_len]
            
    #         # Convert to numeric types if possible
    #         try:
    #             x = [float(val) if not pd.isna(val) else 0 for val in x]
    #             y = [float(val) if not pd.isna(val) else 0 for val in y]
    #         except (ValueError, TypeError):
    #             # If conversion fails, try to use as is
    #             pass
            
    #         # Create the plot based on type
    #         if plot_type == "line":
    #             plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6)
    #         elif plot_type == "bar":
    #             # For bar plots, x should be indices or categories
    #             if isinstance(x[0], str):
    #                 x_pos = range(len(x))
    #                 plt.bar(x_pos, y)
    #                 plt.xticks(x_pos, x, rotation=45, ha='right')
    #             else:
    #                 plt.bar(x, y)
    #         elif plot_type == "scatter":
    #             if label is not None and len(label) > 0:
    #                 try:
    #                     # Try to convert labels to numeric for coloring
    #                     numeric_label = pd.to_numeric(label[:min_len], errors='coerce')
    #                     # Replace NaN with 0
    #                     numeric_label = [0 if pd.isna(val) else val for val in numeric_label]
    #                     scatter = plt.scatter(x, y, c=numeric_label, cmap='viridis', s=50)
    #                     plt.colorbar(scatter, label='Label Values')
    #                 except Exception:
    #                     # If numeric conversion fails, just plot without colors
    #                     plt.scatter(x, y, s=50)
    #             else:
    #                 plt.scatter(x, y, s=50)
    #         else:
    #             plt.close('all')
    #             return f"Error: Unsupported plot_type: {plot_type}. Use 'line', 'bar', or 'scatter'"
    #         plt.show()
    #         # Add labels and title
    #         plt.title(title, fontsize=14, fontweight='bold')
    #         plt.xlabel("X-axis", fontsize=12)
    #         plt.ylabel("Y-axis", fontsize=12)
    #         plt.grid(True, alpha=0.3)
    #         plt.tight_layout()
            
    #         # Save to bytes buffer
    #         buf = io.BytesIO()
    #         plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    #         buf.seek(0)
            
    #         # Encode to base64
    #         img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
    #         # Clean up
    #         buf.close()
    #         plt.close(fig)
    #         plt.close('all')
            
    #         # Return with prefix for easy parsing
    #         return f"base64_image:{img_base64}"
            
    #     except Exception as e:
    #         # Ensure cleanup even on error
    #         try:
    #             plt.close('all')
    #         except:
    #             pass
    #         return f"Error: Failed to generate plot - {str(e)}"

# # Test the function
# result = gen_plot(plot_type="line", title="Test Plot", x=[1, 2, 3, 4, 5], y=[2, 4, 1, 3, 5])
# print(f"Result length: {len(result)}")
# print(f"Result starts with: {result[:50]}...")
# print("\nFunction executed successfully!")
   
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
        plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return 'image created'
