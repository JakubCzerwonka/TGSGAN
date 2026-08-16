from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model import GAN, get_samples, get_preds, get_props_and_error
import json
import joblib

with open('../Transformer/token_to_id.json', 'r') as f:
    token_to_id = json.load(f)
with open('../Transformer/id_to_token.json', 'r') as f:
    id_to_token = json.load(f)

std_sclr = joblib.load('../Transformer/StandardScaler.pkl')
special_tokens = ['[START]', '[END]', '[PAD]']

model = GAN()
model.generator.load_weights('../SavedWeights/generator.weights.h5')

app = FastAPI()
app.mount("/static", StaticFiles(directory='static'), name='static')

@app.get("/", response_class=HTMLResponse)
async def home_page():
    with open("static/index.html", encoding='utf-8') as f:
        return f.read()

@app.post("/", response_class=HTMLResponse)
async def submit_formu(
    A: float = Form(...),
    B: float = Form(...),
    C: float = Form(...),
    Cv: float = Form(...),
    G: float = Form(...),
    G_atomization: float = Form(...),
    H: float = Form(...),
    H_atomization: float = Form(...),
    U: float = Form(...),
    U0: float = Form(...),
    U0_atomization: float = Form(...),
    U_atomization: float = Form(...),
    alpha: float = Form(...),
    gap: float = Form(...),
    homo: float = Form(...),
    lumo: float = Form(...),
    mu: float = Form(...),
    r2: float = Form(...),
    zpve: float = Form(...),
    qed: float = Form(...),
    LogP: float = Form(...),
    TPSA: float = Form(...),
    mw: float = Form(...)
):
    with open('static/index.html', encoding='utf-8') as f:
        html = f.read()

    props = [[A, B, C, Cv, G, G_atomization, H, 
             H_atomization, U, U0, U0_atomization, 
             U_atomization, alpha, gap, homo,
             lumo, mu, r2, zpve, qed, LogP, TPSA, mw]]

    
    preds = get_preds(input_props=props, generator=model.generator, std_sclr=std_sclr, 
                      token_to_id=token_to_id, seq_len=25, id_to_token=id_to_token, 
                      special_tokens=special_tokens)

    add_props, error = get_props_and_error(preds, props[0][-4:])

    html = html.replace(
        "<p>Enter properties...</p>",
        f"<p>Prediction: {preds}</p> <br> QED: {add_props[0]}\
          <br> Log P: {add_props[1]} <br> TPSA: {add_props[2]} \
          <br> Molecular Weight: {add_props[3]} \
          <br><strong>Error (MSE): {error}</strong>"
    )

    return html
