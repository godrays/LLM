#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

import os
import subprocess
import sys
import platform
import shutil


def downloadGPT2Files(modelSize, modelDir):
    import requests, json
    from tqdm import tqdm

    for filename in [
        "checkpoint",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{modelSize}/{filename}", stream=True)
        r.raise_for_status()

        fileFullPath = os.path.join(modelDir, filename)
        with open(fileFullPath, "wb") as f:
            fileSize = int(r.headers["content-length"])
            chunkSize = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=fileSize, unit_scale=True, unit="b") as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes.
                for chunk in r.iter_content(chunk_size=chunkSize):
                    f.write(chunk)
                    pbar.update(chunkSize)


def loadGPT2Params(checkpointPath, hparams):
    import re
    import numpy as np
    import tensorflow as tf

    def setInNestedDict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = setInNestedDict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(checkpointPath):
        array = np.squeeze(tf.train.load_variable(checkpointPath, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            setInNestedDict(params["blocks"][n], sub_name.split("/"), array)
        else:
            setInNestedDict(params, name.split("/"), array)

    return params


def loadHparamsAndParams(modelSize, modelsDir):
    import json
    import tensorflow as tf
    assert modelSize in ["124M", "355M", "774M", "1558M"]

    modelDir = os.path.join(modelsDir, modelSize)
    checkpointPath = tf.train.latest_checkpoint(modelDir)
    if not checkpointPath:
        os.makedirs(modelDir, exist_ok=True)
        downloadGPT2Files(modelSize, modelDir)
        checkpointPath = tf.train.latest_checkpoint(modelDir)

    hparams = json.load(open(os.path.join(modelDir, "hparams.json")))
    params = loadGPT2Params(checkpointPath, hparams)
    return hparams, params


def saveWeights(d, file):
    import numpy as np
    if isinstance(d, np.ndarray):
        file.write(np.uint64(d.size).tobytes())
        d.ravel().tofile(file)
    elif isinstance(d, list):
        return [saveWeights(v, file) for v in d]
    elif isinstance(d, dict):
        return {k: saveWeights(v, file) for k, v in d.items()}
    else:
        ValueError("Unexpected value error.")


def downloadAndSaveWeights(modelSize, modelsDir, targetFilepath):
    hparams, params = loadHparamsAndParams(modelSize, modelsDir)
    with open(targetFilepath, 'wb') as file:
        saveWeights(params, file)


# ----------------------------------------------------------------------------
# HELPER VIRTUAL ENVIRONMENT AND MAIN FUNCTIONS
# ----------------------------------------------------------------------------


def virtualMain(tempDir, modelDir, modelSize):
    # Install the required Python packages into a temporary virtual environment to avoid polluting the system.
    modelFilename = "oaiWeights"
    modelFileExt = ".bin"

    try:
        # Remove previous temporary dir.
        if os.path.exists(tempDir):
            shutil.rmtree(tempDir)
        # Install python packages.
        sourceFilepath = os.path.join(tempDir, modelSize, modelFilename + modelFileExt)
        targetFilepath = os.path.join(modelDir, modelFilename + modelSize + modelFileExt)
        os.makedirs(os.path.dirname(sourceFilepath), exist_ok=True)
        os.makedirs(os.path.dirname(targetFilepath), exist_ok=True)
        # Download all OpenAI GPT2 model weights.
        downloadAndSaveWeights(modelSize, tempDir, targetFilepath)
    finally:
        # Cleanup.
        shutil.rmtree(tempDir)


def initVirtualEnvironment(venvName, venvPackages):
    # Determine os specific commands.
    if platform.system() == "Windows":
        activate_cmd = f"{venvName}\\Scripts\\activate"
        separator = " & "   # Use & to chain on Windows
        removeVenv = f"rmdir {venvName} /s /q"
    else:
        activate_cmd = f"source {venvName}/bin/activate"
        separator = " && "  # Use && to chain on POSIX
        removeVenv = f"rm -rf {venvName}"

    # Construct the full command string.
    fullCommand = (
        f"python -c \"import sys; print('Running Python: ', sys.version)\"{separator}"
        f"python -m venv {venvName}{separator}"
        f"{activate_cmd}{separator}"
        f"pip install --upgrade pip{separator}"
        f"pip install {' '.join(venvPackages)}{separator}"
        f"python {sys.argv[0]} run {sys.argv[2]}{separator}"
        f"{removeVenv}"
    )

    # Run the commands in a single shell process.
    subprocess.run(fullCommand, check=True, shell=True, capture_output=False)


if __name__ == "__main__":
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print("Please use python version 3.11.x. (Use pyenv to install)")
        exit()

    if (len(sys.argv) != 3
            or (not sys.argv[1] in ["init", "run"])
            or (not sys.argv[2] in ["124M", "355M", "774M", "1558M"])):
        print("Please provide a model size to download as an argument. Options: 124M, 355M, 774M, 1558M")
        print("Example:")
        print("         python " + sys.argv[0] + " init 124M")
        exit()

    mode = sys.argv[1]
    modelType = sys.argv[2]

    # Packages that will be installed in the virtual environment.
    venvPackages = ["numpy==2.1.3",
                    "regex==2024.11.6",
                    "requests==2.32.3",
                    "tqdm==4.67.1",
                    "urllib3==2.3.0",
                    "tensorflow==2.19.0"]

    if mode == "init":      # The script will initiate virtual environment.
        initVirtualEnvironment(venvName="temp_venv", venvPackages=venvPackages)
    elif mode == "run":       # The script will call itself inside venv with run parameter.
        virtualMain(tempDir="temp_dir", modelDir="GPT2", modelSize=modelType)
