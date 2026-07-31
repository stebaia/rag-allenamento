#!/bin/bash
# Crea lo zip di deployment per la Lambda della skill Alexa: installa le
# dipendenze (ask-sdk-core, ask-sdk-model, requests) in una cartella
# temporanea insieme al codice, e comprime tutto in build/lambda.zip,
# pronto da caricare sulla console AWS Lambda ("Upload from" > ".zip file").
#
# Uso: dalla cartella alexa-skill/, eseguire ./build.sh
set -euo pipefail

cd "$(dirname "$0")"

rm -rf build
mkdir -p build/package

pip install -r requirements.txt --target build/package --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.13

cp lambda_function.py build/package/

cd build/package
zip -r ../lambda.zip . -x "*.pyc" "*__pycache__*"
cd ../..

echo "Creato alexa-skill/build/lambda.zip - caricalo su AWS Lambda"
