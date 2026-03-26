#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from cgm_diabetes.data.CGMDiabetesDataset import CGMDiabetesDataset
import sys

split = sys.argv[1] if len(sys.argv) > 1 else None
CGMDiabetesDataset.prefetch(split)