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


def createVirtualEnv(venv_name):
    subprocess.check_call([sys.executable, "-m", "venv", venv_name])


def installPackages(pipBin, packages):
    subprocess.check_call([pipBin, "install"] + packages)


def getVirtualEnvPaths(venvName):
    if os.name == 'nt':     # windows
        pipBin = os.path.join(venvName, "Scripts", "pip")
    else:                   # macOS, unix, linux
        pipBin = os.path.join(venvName, "bin", "pip")
    return pipBin


def main():
    if (sys.version_info.major != 3 or sys.version_info.minor != 11):
        print("This script requires python version 3.11.x")
        exit()

    if len(sys.argv) != 2:
        print("Please provide a model size to download as an argument. Options: 124M, 355M, 774M, 1558M")
        print("Example:")
        print("         python " + sys.argv[0] + " 124M")
        exit()

    # Install the required Python packages into a temporary virtual environment to avoid polluting the system.
    packages = ["numpy", "regex", "requests", "tqdm", "urllib3", "tensorflow"]
    venvDir = "temp_venv"
    tempDir = "./tempdir"
    modelDir = "./GPT2"
    modelSizes = ["124M", "355M", "774M", "1558M"]
    modelFilename = "oaiWeights"
    modelFileExt = ".bin"
    modelSize =  sys.argv[1]

    assert modelSize in modelSizes

    createVirtualEnv(venvDir)
    pipBin = getVirtualEnvPaths(venvDir)

    try:
        # Remove previous temporary dir.
        if os.path.exists(tempDir):
            shutil.rmtree(tempDir)
        # Install python packages.
        installPackages(pipBin, packages)
        sourceFilepath = os.path.join(tempDir, modelSize, modelFilename + modelFileExt)
        targetFilepath = os.path.join(modelDir, modelFilename + modelSize + modelFileExt)
        os.makedirs(os.path.dirname(sourceFilepath), exist_ok=True)
        os.makedirs(os.path.dirname(targetFilepath), exist_ok=True)
        # Download all OpenAI GPT2 model weights.
        downloadAndSaveWeights(modelSize, tempDir, targetFilepath)
    finally:
        # Cleanup.
        shutil.rmtree(venvDir)
        shutil.rmtree(tempDir)


if __name__ == "__main__":
    main()
