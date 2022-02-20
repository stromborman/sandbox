#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.dat', delim_whitespace=True, header=None)

df.dtypes
