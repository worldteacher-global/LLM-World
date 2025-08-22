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

import json

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
    base64_string: str

# class PlotResponse(BaseModel):
#     base64_string: constr(min_length=10) = Field(
#         ..., description="Base64-encoded PNG image string."
#     )
#     file_path: Optional[str] = Field(
#         None, description="Optional file path where the PNG image is saved."
#     )



class DataFramePayload(BaseModel):
    columns: List[str]
    data: List[List[float]]

class MyTools:
####################################################### TOOLS ############################################################

    @tool
    @staticmethod
    def calculate_PC(path: str | None = None):
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
    def levene_test(path: str | None = None):
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
    def anova_one_way(path: str | None = None):
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
    def covariance_tool(path: str | None = None):
        '''Calculate covariance'''
        df = pd.read_csv(path)
        cov = df.cov()
        return DataFramePayload(
            columns=df.columns.tolist(),
            data=cov.values.tolist())

    @tool
    @staticmethod
    def correlation_tool(path: str | None = None):
        '''Calculate correlation'''
        df = pd.read_csv(path)
        corr = df.corr()
        return DataFramePayload(
            columns=df.columns.tolist(),
            data=corr.values.tolist())
        
    @tool
    @staticmethod
    def normality_tool(path: str | None = None):
        '''Calculate if the data is normal'''
        df = pd.read_csv(path)
        for col in df.columns:
            tstat, pval = shapiro(df[col])
            if pval < 0.05:
                return {'result':f'Column {col} is not normally distributed, and not recommended to include in ANOVA analysis'}
        return {'result':'Data is normal'}




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
        # plt.show()
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
        # with open(filepath, "rb") as img_file:
        #     img_base64 = base64.b64encode(img_file.read()).decode("utf-8")        
        # print(f"Plot saved to: {filepath}")
  
        # return PlotResponse(base64_string=img_base64)
        return {"output":filepath} ## Worked!!
        # return {"output":img_base64} ## Worked!!
        # return json.dumps({"return":img_base64},indent=3)
  
        # return PlotResponse(base64_string=img_base64, filepath=filepath)
    
    @tool
    @staticmethod
    def display_base64_image(filepath: str | None = None,
        base64_string: str | None = None,
        caption: str = "Image"):
        """Displays an image from either a file path, base64-encoded string, or reads from directory."""

        if isinstance(filepath, dict) and "filepath" in filepath:
            file_path = filepath["filepath"]
        else:
            file_path = filepath

        # Load from file path
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            st.image(file_path, caption=caption)
            # img = Image.open(file_path)
            # img.show()
            return {"status": "ok", "caption": caption, "filepath": file_path}

        # Load from base64
        if base64_string:
            if base64_string.startswith("base64_image:"):
                base64_string = base64_string[13:]

            if base64_string.startswith("data:image/png;base64,"):
                base64_string = base64_string.split(",", 1)[1]

            # image_bytes = base64.b64decode(base64_string)
            image_bytes = base64.b64decode(base64_string.encode("utf-8"))
            # image = Image.open(io.BytesIO(image_bytes))
            st.image(image_bytes, caption=caption)
            return {"status": "ok", "caption": caption}

        return {"status": "error", "message": "No image found"}
