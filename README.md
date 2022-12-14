# Welcome to Fabrice's Project 7!

## Install Anaconda

Follow the steps at the [Anaconda installation page](https://docs.anaconda.com/anaconda/install/windows/).

## Create a new environment with Streamlit

1. Start Anaconda "Getting Started" page
2. Click the "▶" icon next to your new environment (root or sth else) and then "Open terminal":

<img src="https://camo.githubusercontent.com/2e4ec1070ef05f008db123248dc0ac9d510f2c43c0a4ab06db147e056f1cb00d/68747470733a2f2f692e737461636b2e696d6775722e636f6d2f45696946632e706e67" width="400"/>

3. In the terminal that appears, type:
```bash
pip install streamlit
```

4. Finish the installation. Copy paste the following code:
```bash
git clone https://github.com/fab83200/project7.git
cd project7
pip install -r requirements.txt
```
_NOTA: All needed libraries to execute the Python code should be included in the file `requirements.txt` in [my repository](https://github.com/fab83200/project7)_.

## Edit the app to deploy

* Edit `/local_app.py` to customize this app to your heart's desire. :heart:

## Deploy the app

To deploy the app, open this [link](https://share.streamlit.io).

If P7 isn't in the list, click on the button `New App` on the right side. Then choose:
Repository:
    fab83200/P7
Branch:
    main
Main file path:
    local_app.py

Finally, click on `Deploy`.

**Nota:** _Sometimes after `Deploy`, it doesn't load right away. Usually a simple `refresh page` F5/Ctrl+R suffice._
