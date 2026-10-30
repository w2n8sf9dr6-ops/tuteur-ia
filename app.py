import gzip
import json
import random
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

st.set_page_config(page_title="Mon espace d'apprentissage", page_icon="🚀")

# =====================================================================
# CONFIGURATION (tout vient des "Secrets" de Streamlit)
# =====================================================================
S = st.secrets
MODEL = S.get("MODEL", "gemini-2.5-flash")
EMBED_MODEL = S.get("EMBED_MODEL", "gemini-embedding-001")
SHEET_URL = S["SHEET_URL"]
SHEET_TOKEN = S["SHEET_TOKEN"]
TEACHER_PASSWORD = S.get("TEACHER_PASSWORD", "")
RELEVANCE = float(S.get("RELEVANCE", 0.5))  # seuil de pertinence de la recherche dans le cours
CONTACT_FORMATEUR = S.get("CONTACT_FORMATEUR", "(contact du formateur à renseigner)")
CONTACT_REFERENT = S.get("CONTACT_REFERENT", "(contact du référent du centre à renseigner)")
LIEN_FORUM = S.get("LIEN_FORUM", "(lien du forum ou du groupe d'étude à renseigner)")

client = genai.Client(api_key=S["GEMINI_API_KEY"])

with open("prompt.txt", encoding="utf-8") as f:
    BASE_PROMPT = f.read()

RAG_RULES = """

# S'APPUYER SUR LE COURS
À chaque question de l'apprenant, l'application peut te fournir des « EXTRAITS DU COURS ».
- Appuie-toi sur ces extraits pour expliquer, et cite la source sous la forme (fichier, p. X).
- Si les extraits ne couvrent pas la question, dis-le honnêtement (« ce point ne semble pas traité dans le support »), n'invente rien, et invite l'apprenant à poser la question à son formateur.
- Garde ta posture de tuteur : guide par questions et indices, sans recopier les extraits.
"""

# =====================================================================
# STOCKAGE : Google Sheets via Apps Script
# =====================================================================
COLS = {
    "sessions": ["code", "session", "type", "energie", "stress", "minutes"],
    "messages": ["code", "session", "role", "texte"],
    "fiches": ["code", "fiche"],
    "quiz": ["code", "session", "notion", "question", "correct", "source"],
    "plans": ["code", "objectif", "etapes", "quand", "si_alors"],
    "pauses": ["code", "type"],
    "unanswered": ["code", "question"],
}


@st.cache_data(ttl=45, show_spinner=False)
def read(sheet, code=None):
    cols = ["date"] + COLS[sheet]
    try:
        params = {"token": SHEET_TOKEN, "sheet": sheet}
        if code:
            params["code"] = code
        rows = requests.get(SHEET_URL, params=params, timeout=25).json().get("rows", [])
    except Exception:
        rows = []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.reindex(columns=range(len(cols)))
    df.columns = cols
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


def log(sheet, row):
    try:
        requests.post(SHEET_URL, json={"token": SHEET_TOKEN, "sheet": sheet, "row": row}, timeout=25)
        read.clear()
    except Exception:
        pass  # on ne bloque jamais l'apprenant si l'enregistrement échoue


# =====================================================================
# GEMINI
# =====================================================================
def to_contents(history):
    return [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["text"])])
        for m in history
    ]


def gen(system, contents, json_mode=False, temperature=0.7):
    if isinstance(contents, str):
        contents = [types.Content(role="user", parts=[types.Part(text=contents)])]
    kwargs = dict(system_instruction=system, temperature=temperature)
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    for attempt in range(2):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=contents, config=types.GenerateContentConfig(**kwargs)
            )
            return r.text or ""
        except Exception:
            if attempt == 1:
                raise
            time.sleep(3)


# =====================================================================
# RECHERCHE DANS LE COURS (index construit par build_index.py)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_index():
    try:
        with gzip.open("index/chunks.json.gz", "rt", encoding="utf-8") as f:
            data = json.load(f)
        emb = np.load("index/embeddings.npy").astype(np.float32)
        return data, emb
    except Exception:
        return None, None


def has_index():
    return load_index()[0] is not None


def embed_query(text, dims):
    r = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dims),
    )
    v = np.array(r.embeddings[0].values, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def retrieve(query, k=4):
    data, emb = load_index()
    if data is None:
        return []
    try:
        q = embed_query(query, data["meta"]["dims"])
    except Exception:
        return []
    sims = emb @ q
    top = np.argsort(-sims)[:k]
    return [
        dict(src=data["src"][i], page=data["page"][i], text=data["text"][i], score=float(sims[i]))
        for i in top
    ]


def random_passages(k=3):
    data, _ = load_index()
    if data is None:
        return []
    n = len(data["text"])
    i = random.randrange(n)
    out = []
    for j in range(i, min(i + k, n)):
        if data["src"][j] == data["src"][i]:
            out.append(dict(src=data["src"][j], page=data["page"][j], text=data["text"][j], score=1.0))
    return out


def passages_block(ps):
    return "\n\n".join(f"[{n}] ({p['src']}, p. {p['page']})\n{p['text']}" for n, p in enumerate(ps, 1))


def src_label(p):
    return f"{p['src']}, p. {p['page']}"


# =====================================================================
# ÉTAT
# =====================================================================
s = st.session_state
s.setdefault("tstep", "checkin")
s.setdefault("history", [])
s.setdefault("qid", 0)
s.setdefault("q", None)
s.setdefault("q_done", False)

NO_INDEX_MSG = (
    "Le support de cours n'est pas encore chargé dans l'application : "
    "cette fonction n'est pas disponible (ou fonctionne sans le cours)."
)


# =====================================================================
# PAGE : TABLEAU DE BORD
# =====================================================================
def notion_stats(quiz):
    cols = ["notion", "n", "taux", "dernier", "rattrapee"]
    if quiz.empty:
        return pd.DataFrame(columns=cols)
    q = quiz.copy()
    q["correct"] = pd.to_numeric(q["correct"], errors="coerce").fillna(0)
    q = q.sort_values("date")
    rows = []
    for notion, g in q.groupby("notion"):
        c = g["correct"].tolist()
        rows.append(
            dict(
                notion=notion,
                n=len(c),
                taux=sum(c) / len(c),
                dernier=bool(c[-1]),
                rattrapee=bool(c[-1]) and (0 in c[:-1]),
            )
        )
    return pd.DataFrame(rows, columns=cols)


def page_dashboard():
    st.header("🏠 Mon tableau de bord")
    sess = read("sessions", s.code)
    quiz = read("quiz", s.code)
    fiches = read("fiches", s.code)

    for c in ("energie", "stress"):
        sess[c] = pd.to_numeric(sess[c], errors="coerce")
    debut = sess[sess["type"] == "debut"].sort_values("date")
    now = pd.Timestamp.now(tz="UTC")

    c1, c2, c3 = st.columns(3)
    c1.metric("Séances réalisées", len(debut))
    recent = debut[debut["date"] >= now - pd.Timedelta(days=28)]
    c2.metric("Semaines actives (sur 4)", int(recent["date"].dt.isocalendar().week.nunique()) if len(recent) else 0)
    qc = pd.to_numeric(quiz["correct"], errors="coerce")
    c3.metric("Réussite aux quiz", f"{int(qc.mean() * 100)} %" if len(qc) else "–")

    if debut.empty:
        st.info("Ton tableau de bord se remplira dès ta première séance avec le tuteur. 🚀")
        return

    st.subheader("Mon énergie et mon stress")
    st.line_chart(debut.set_index("date")[["energie", "stress"]])
    st.caption("Repère : de petites séances régulières valent mieux qu'une longue séance épuisante.")

    stats = notion_stats(quiz)
    if len(stats):
        victoires = stats[stats["rattrapee"]]
        if len(victoires):
            st.subheader("🎉 Mes petites victoires")
            for n in victoires["notion"]:
                st.success(f"Tu as rattrapé « {n} » : tu t'étais trompé(e) avant, et tu as réussi ensuite !")
        fragiles = stats[(~stats["dernier"]) | (stats["taux"] < 0.6)]
        if len(fragiles):
            st.subheader("🎯 Notions à retravailler")
            for n in fragiles["notion"]:
                st.write(f"- {n}")
            st.caption("Va dans « Je me teste » : le bouton « Réviser mes points fragiles » est fait pour ça.")

    if len(fiches):
        with st.expander("Ma dernière fiche de séance"):
            st.markdown(fiches.iloc[-1]["fiche"])


# =====================================================================
# PAGE : TUTEUR
# =====================================================================
def start_session(energy, stress, minutes):
    fdf = read("fiches", s.code)
    fiche = fdf.iloc[-1]["fiche"] if len(fdf) else ""
    s.energy0, s.stress0 = energy, stress
    s.session_id = uuid.uuid4().hex[:8]
    s.system = (
        BASE_PROMPT
        + f"""

# CONTEXTE DE CETTE SÉANCE (fourni par l'application)
- Énergie : {energy}/5 ; stress : {stress}/5 ; temps disponible : {minutes} min.
- Fiche de la séance précédente :
{fiche if fiche else 'Aucune : première séance.'}
Le check-in est déjà fait : ne le redemande pas. Salue, rappelle brièvement où l'apprenant en était (si une fiche existe), puis aide-le à fixer UN objectif réaliste pour cette séance.
"""
        + RAG_RULES
    )
    log("sessions", [s.code, s.session_id, "debut", energy, stress, minutes])
    s.history = [{"role": "user", "text": "Bonjour, je commence ma séance."}]
    s.history.append({"role": "model", "text": gen(s.system, to_contents(s.history))})
    s.tstep = "chat"


def page_tuteur():
    st.header("💬 Mon tuteur")

    if s.tstep == "checkin":
        st.write("Avant de commencer, trois questions rapides pour adapter la séance.")
        energy = st.slider("Ton énergie aujourd'hui (1 = très basse, 5 = très haute)", 1, 5, 3)
        stress = st.slider("Ton niveau de stress (1 = très bas, 5 = très élevé)", 1, 5, 3)
        minutes = st.number_input("Temps dont tu disposes (minutes)", 10, 240, 45, step=5)
        if not has_index():
            st.warning(NO_INDEX_MSG)
        if st.button("Lancer la séance"):
            with st.spinner("Je prépare ta séance..."):
                start_session(energy, stress, int(minutes))
            st.rerun()

    elif s.tstep == "chat":
        for m in s.history[1:]:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.write(m["text"])
                if m.get("src"):
                    st.caption("📚 Voir dans le cours : " + " · ".join(m["src"]))

        if prompt := st.chat_input("Écris ton message..."):
            s.history.append({"role": "user", "text": prompt})
            log("messages", [s.code, s.session_id, "apprenant", prompt])
            extra, sources = "", []
            if has_index():
                ps = retrieve(prompt, k=4)
                good = [p for p in ps if p["score"] >= RELEVANCE]
                if good:
                    extra = "\n\n# EXTRAITS DU COURS (pour la dernière question)\n" + passages_block(good)
                    sources = list(dict.fromkeys(src_label(p) for p in good[:3]))
                else:
                    extra = (
                        "\n\n# EXTRAITS DU COURS\nAucun extrait pertinent trouvé : dis honnêtement que ce point "
                        "ne semble pas traité dans le support, et suggère de poser la question au formateur."
                    )
                    log("unanswered", [s.code, prompt[:300]])
            with st.spinner("Le tuteur réfléchit..."):
                try:
                    answer = gen(s.system + extra, to_contents(s.history))
                except Exception:
                    answer = "Petit souci technique, peux-tu renvoyer ton message dans quelques secondes ?"
            s.history.append({"role": "model", "text": answer, "src": sources})
            log("messages", [s.code, s.session_id, "tuteur", answer])
            st.rerun()

        st.divider()
        if st.button("✅ Terminer la séance"):
            s.tstep = "bilan"
            st.rerun()

    elif s.tstep == "bilan":
        st.subheader("Bilan de séance")
        energy1 = st.slider("Ton énergie maintenant (1 à 5)", 1, 5, 3)
        stress1 = st.slider("Ton stress maintenant (1 à 5)", 1, 5, 3)
        learned = st.text_area("Qu'as-tu compris ou appris aujourd'hui ?")
        blocked = st.text_area("Où as-tu bloqué, et qu'est-ce qui t'a aidé ?")
        nxt = st.text_input("Ta prochaine étape, et quand ?")
        if st.button("Générer ma fiche de séance") and learned:
            demande = (
                "La séance est terminée. Voici le bilan de l'apprenant.\n"
                f"Appris : {learned}\nBlocages et aides : {blocked}\nProchaine étape : {nxt}\n"
                f"Énergie/stress début : {s.energy0}/{s.stress0} ; fin : {energy1}/{stress1}.\n"
                "Génère maintenant la FICHE DE SÉANCE (format du prompt), courte, en listant les notions "
                "maîtrisées et fragiles d'après la conversation. Ajoute une phrase d'encouragement précise."
            )
            with st.spinner("Je prépare ta fiche..."):
                fiche = gen(s.system, to_contents(s.history + [{"role": "user", "text": demande}]))
            log("fiches", [s.code, f"[{datetime.now().strftime('%d/%m/%Y')}] {fiche}"])
            log("sessions", [s.code, s.session_id, "fin", energy1, stress1, ""])
            s.fiche_text = fiche
            s.tstep = "fin"
            st.rerun()

    elif s.tstep == "fin":
        st.success("Fiche enregistrée. Elle sera rechargée automatiquement à ta prochaine séance.")
        st.markdown(s.fiche_text)
        if st.button("Démarrer une nouvelle séance"):
            s.tstep, s.history = "checkin", []
            st.rerun()


# =====================================================================
# PAGE : JE ME TESTE
# =====================================================================
QUIZ_SYS = """Tu crées des questions d'entraînement (QCM) pour des adultes en formation, à partir UNIQUEMENT des extraits de cours fournis.
Règles : une seule bonne réponse, trois distracteurs plausibles, question claire et utile pour la pratique professionnelle, niveau demandé respecté.
Réponds uniquement par un objet JSON de la forme :
{"notion": "nom court de la notion (2 à 5 mots)", "question": "...", "options": ["...", "...", "...", "..."], "bonne": 0, "explication": "2 à 3 phrases pédagogiques", "extrait": 1}
où "bonne" est l'indice (0 à 3) de la bonne option et "extrait" le numéro de l'extrait utilisé."""


def new_question(passages, niveau):
    prompt = f"Niveau : {niveau}.\n\nEXTRAITS :\n{passages_block(passages)}"
    for _ in range(2):
        try:
            raw = gen(QUIZ_SYS, prompt, json_mode=True, temperature=0.6)
            d = json.loads(raw)
            if isinstance(d, list):
                d = d[0]
            opts = [str(o) for o in d["options"]]
            assert len(opts) == 4 and 0 <= int(d["bonne"]) <= 3
            bonne_txt = opts[int(d["bonne"])]
            random.shuffle(opts)
            idx = min(max(int(d.get("extrait", 1)), 1), len(passages)) - 1
            s.qid += 1
            s.q = dict(
                notion=str(d["notion"])[:60],
                question=str(d["question"]),
                options=opts,
                bonne=opts.index(bonne_txt),
                explication=str(d["explication"]),
                source=src_label(passages[idx]),
            )
            s.q_done = False
            return True
        except Exception:
            continue
    return False


def page_quiz():
    st.header("🧠 Je me teste")
    st.caption("Se tester sans regarder le cours est l'une des méthodes les plus efficaces pour retenir.")
    if not has_index():
        st.info(NO_INDEX_MSG)
        return

    niveau = st.select_slider("Niveau", ["facile", "moyen", "difficile"], value="moyen")
    topic = st.text_input("Un sujet précis ? (sinon, laisse vide pour une question au hasard)")
    c1, c2 = st.columns(2)
    go = c1.button("Nouvelle question")
    review = c2.button("🎯 Réviser mes points fragiles")

    if go or review:
        passages = []
        if review:
            stats = notion_stats(read("quiz", s.code))
            frag = stats[~stats["dernier"]] if len(stats) else stats
            if len(frag):
                topic = random.choice(frag["notion"].tolist())
                st.info(f"On révise : {topic}")
            else:
                st.info("Tu n'as pas encore de point fragile : voici une question au hasard.")
        if topic:
            passages = retrieve(topic, k=3)
            if not passages or passages[0]["score"] < RELEVANCE - 0.1:
                st.warning("Je ne trouve pas ce sujet dans le cours. Essaie avec d'autres mots.")
                passages = []
        else:
            passages = random_passages(3)
        if passages:
            with st.spinner("Je prépare ta question..."):
                if not new_question(passages, niveau):
                    st.error("Je n'ai pas réussi à créer la question, réessaie.")

    q = s.q
    if q:
        st.divider()
        st.markdown(f"**{q['question']}**")
        choice = st.radio("Ta réponse", q["options"], index=None, key=f"choice_{s.qid}", disabled=s.q_done)
        if not s.q_done and st.button("Valider ma réponse") and choice is not None:
            ok = q["options"].index(choice) == q["bonne"]
            s.q_ok = ok
            s.q_done = True
            log("quiz", [s.code, s.get("session_id", ""), q["notion"], q["question"][:200], int(ok), q["source"]])
            st.rerun()
        if s.q_done:
            if s.q_ok:
                st.success("Bravo, c'est la bonne réponse ! ✅")
            else:
                st.error(f"Pas tout à fait. La bonne réponse : {q['options'][q['bonne']]}")
            st.write(q["explication"])
            st.caption(f"📚 Pour revoir : {q['source']}  ·  Notion : {q['notion']}")


# =====================================================================
# PAGE : MON PLAN
# =====================================================================
def page_plan():
    st.header("📅 Mon plan")
    tab1, tab2 = st.tabs(["Mon plan de la semaine", "Je suis submergé(e)"])

    with tab1:
        prev = read("plans", s.code)
        if len(prev):
            p = prev.iloc[-1]
            st.info(f"**Dernier plan enregistré** : {p['objectif']}\n\nÉtapes : {p['etapes']}\n\nQuand : {p['quand']}")
        objectif = st.text_input("Mon objectif de la semaine (concret et réaliste)")
        e1 = st.text_input("Étape 1")
        e2 = st.text_input("Étape 2")
        e3 = st.text_input("Étape 3")
        quand = st.text_input("Quand ? (jours et horaires précis)")
        obstacle = st.text_input("Ce qui pourrait m'empêcher d'avancer")
        solution = st.text_input("Si cela arrive, alors je...")
        c1, c2 = st.columns(2)
        if c1.button("Enregistrer mon plan") and objectif:
            etapes = " / ".join(x for x in (e1, e2, e3) if x)
            log("plans", [s.code, objectif, etapes, quand, f"Si {obstacle}, alors {solution}"])
            st.success("Plan enregistré. Bravo d'avoir pris le temps de t'organiser ! 👏")
        if c2.button("💡 Améliorer mon plan") and objectif:
            demande = (
                f"Objectif : {objectif}\nÉtapes : {e1} / {e2} / {e3}\nQuand : {quand}\n"
                f"Obstacle : {obstacle} -> {solution}"
            )
            with st.spinner("J'analyse ton plan..."):
                st.markdown(
                    gen(
                        "Tu es un coach d'organisation bienveillant pour adultes en formation à distance. "
                        "Analyse ce plan : dis ce qui est bien, signale s'il est trop ambitieux ou trop vague, "
                        "propose 2 améliorations concrètes. Réponse courte (moins de 150 mots), en français.",
                        demande,
                    )
                )

    with tab2:
        st.write("Quand tout s'accumule, on ne cherche pas à tout faire : on trie et on allège.")
        taches = st.text_area("Liste tout ce que tu as à faire (une chose par ligne)")
        heures = st.number_input("Heures réellement disponibles cette semaine", 1, 40, 5)
        if st.button("M'aider à trier") and taches.strip():
            with st.spinner("Je trie avec toi..."):
                rep = gen(
                    "Tu es un coach bienveillant. L'apprenant est submergé. Classe ses tâches en 3 groupes "
                    "(Urgent / Important / Peut attendre), propose un objectif de la semaine ALLÉGÉ et réaliste "
                    "pour son temps disponible, et UNE petite action à faire dès maintenant (moins de 15 minutes). "
                    "Ton rassurant, sans pression, moins de 200 mots, en français.",
                    f"Heures disponibles : {heures}\nTâches :\n{taches}",
                )
            st.markdown(rep)
            log("pauses", [s.code, "submerge"])


# =====================================================================
# PAGE : MA PAUSE
# =====================================================================
TIMER_HTML = """
<div style="font-family:sans-serif;text-align:center;background:#f4f6f8;color:#222;border-radius:12px;padding:16px">
  <div id="t" style="font-size:52px;font-weight:600">25:00</div>
  <div style="margin-top:8px">
    <button onclick="start(25)" style="padding:8px 14px;margin:3px">Étude 25 min</button>
    <button onclick="start(5)" style="padding:8px 14px;margin:3px">Pause 5 min</button>
    <button onclick="stopT()" style="padding:8px 14px;margin:3px">Stop</button>
  </div>
</div>
<script>
let iv, left = 25*60;
function show(){const m=Math.floor(left/60), s=left%60;
 document.getElementById('t').textContent=String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');}
function start(min){clearInterval(iv); left=min*60; show();
 iv=setInterval(()=>{left--; show(); if(left<=0){clearInterval(iv); document.getElementById('t').textContent='Terminé ✅';}},1000);}
function stopT(){clearInterval(iv); left=25*60; show();}
</script>
"""

BREATH_HTML = """
<style>
.box{font-family:sans-serif;text-align:center;background:#eef5f1;color:#234;border-radius:12px;padding:20px}
.circle{width:120px;height:120px;border-radius:50%;background:#7fc4a0;margin:20px auto;animation:breathe 10s ease-in-out infinite}
@keyframes breathe{0%{transform:scale(1)}40%{transform:scale(1.7)}100%{transform:scale(1)}}
.l{position:absolute;left:0;right:0;font-size:22px;opacity:0}
.in{animation:fin 10s infinite}.out{animation:fout 10s infinite}
@keyframes fin{0%{opacity:1}38%{opacity:1}42%{opacity:0}100%{opacity:0}}
@keyframes fout{0%{opacity:0}40%{opacity:0}44%{opacity:1}98%{opacity:1}100%{opacity:0}}
</style>
<div class="box">
  <div style="position:relative;height:30px"><div class="l in">Inspire doucement (4 s)</div><div class="l out">Expire lentement (6 s)</div></div>
  <div class="circle"></div>
  <div>Suis le cercle pendant 2 minutes. Épaules relâchées, regard doux.</div>
</div>
"""


def page_pause():
    st.header("🌿 Ma pause")
    tab1, tab2, tab3 = st.tabs(["⏱️ Minuteur", "🫁 Respirer", "💡 Mini-pauses"])
    with tab1:
        st.write("Méthode simple : 25 minutes de travail concentré, puis 5 minutes de vraie pause.")
        components.html(TIMER_HTML, height=170)
    with tab2:
        components.html(BREATH_HTML, height=290)
        if st.button("J'ai fait mon exercice de respiration"):
            log("pauses", [s.code, "respiration"])
            st.success("Bien joué, prendre soin de soi fait partie de l'apprentissage. 🌱")
    with tab3:
        st.markdown(
            "- **20-20-20** : toutes les 20 min, regarde à 6 mètres pendant 20 secondes.\n"
            "- **Bouge** : debout, étire la nuque, les épaules et le dos pendant 2 minutes.\n"
            "- **Hydrate-toi** : un verre d'eau, loin de l'écran.\n"
            "- **Fenêtre** : 3 minutes de lumière naturelle ou d'air frais.\n"
            "- **Pense à une victoire** : une chose que tu as comprise cette semaine.\n"
        )
        st.info(
            "Tu es fatigué(e) ou stressé(e) aujourd'hui ? Lance une séance avec le tuteur et indique-le au "
            "check-in : il allégera l'objectif et découpera en petites étapes. Mieux vaut 15 minutes sereines "
            "qu'une heure subie."
        )
        if st.button("J'ai pris une mini-pause"):
            log("pauses", [s.code, "mini-pause"])
            st.success("Parfait, ton cerveau te remerciera. ✅")


# =====================================================================
# PAGE : BESOIN D'UN HUMAIN ?
# =====================================================================
def page_humain():
    st.header("🤝 Besoin d'un humain ?")
    st.write(
        "Le tuteur IA est là pour t'aider à apprendre, mais il ne remplace ni ton formateur, "
        "ni tes pairs, ni un professionnel."
    )
    st.markdown(
        f"- **Une question sur le cours, une difficulté, une décision à prendre** : {CONTACT_FORMATEUR}\n"
        f"- **Une situation compliquée (charge, organisation, handicap, soucis personnels qui pèsent)** : {CONTACT_REFERENT}\n"
        f"- **Échanger avec d'autres apprenants** : {LIEN_FORUM}\n"
    )
    st.warning(
        "Si tu te sens très mal, épuisé(e) ou en détresse, parles-en à quelqu'un dès aujourd'hui. "
        "**3114** : numéro national de prévention du suicide (gratuit, 24h/24). "
        "**15** (SAMU) en cas de danger immédiat."
    )


# =====================================================================
# PAGE : ESPACE FORMATEUR (lecture seule, rien à remplir)
# =====================================================================
def page_formateur():
    st.title("👩‍🏫 Espace formateur")
    if not TEACHER_PASSWORD:
        st.error("Le mot de passe formateur n'est pas configuré (secret TEACHER_PASSWORD).")
        return
    if not s.get("teacher_ok"):
        pw = st.text_input("Mot de passe", type="password")
        if st.button("Entrer"):
            if pw == TEACHER_PASSWORD:
                s.teacher_ok = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        return

    st.caption("Page en lecture seule : tout est calculé automatiquement, il n'y a rien à remplir.")
    sess = read("sessions")
    quiz = read("quiz")
    unans = read("unanswered")
    for c in ("energie", "stress"):
        sess[c] = pd.to_numeric(sess[c], errors="coerce")
    debut = sess[sess["type"] == "debut"].sort_values("date")
    now = pd.Timestamp.now(tz="UTC")

    c1, c2, c3 = st.columns(3)
    c1.metric("Apprenants actifs (7 jours)", debut[debut["date"] >= now - pd.Timedelta(days=7)]["code"].nunique())
    c2.metric("Séances (7 jours)", int((debut["date"] >= now - pd.Timedelta(days=7)).sum()))
    c3.metric("Apprenants au total", debut["code"].nunique())

    st.subheader("Suivi par apprenant (codes anonymes)")
    rows = []
    for code, g in debut.groupby("code"):
        last = g["date"].max()
        jours = (now - last).days if pd.notna(last) else None
        recent_e = g.tail(2)["energie"].mean()
        qq = quiz[quiz["code"] == code]
        taux = pd.to_numeric(qq["correct"], errors="coerce").mean() if len(qq) else np.nan
        flags = []
        if jours is not None and jours >= 14:
            flags.append("inactif depuis 14 j ou plus")
        if pd.notna(recent_e) and recent_e <= 2:
            flags.append("énergie basse")
        rows.append(
            {
                "Code": code,
                "Séances": len(g),
                "Dernière séance (jours)": jours,
                "Énergie récente (1-5)": round(recent_e, 1) if pd.notna(recent_e) else None,
                "Réussite quiz": f"{int(taux * 100)} %" if pd.notna(taux) else "–",
                "À surveiller": ", ".join(flags),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("À surveiller", ascending=False), hide_index=True)
    else:
        st.info("Aucune séance pour le moment.")

    st.subheader("Notions les plus ratées par le groupe")
    if len(quiz):
        q = quiz.copy()
        q["correct"] = pd.to_numeric(q["correct"], errors="coerce").fillna(0)
        agg = q.groupby("notion")["correct"].agg(["count", "mean"]).reset_index()
        agg["Taux d'échec"] = ((1 - agg["mean"]) * 100).round().astype(int).astype(str) + " %"
        agg = agg[agg["count"] >= 2].sort_values("mean").head(15)
        st.dataframe(
            agg.rename(columns={"notion": "Notion", "count": "Questions"})[["Notion", "Questions", "Taux d'échec"]],
            hide_index=True,
        )
    else:
        st.info("Pas encore de données de quiz.")

    st.subheader("Questions sans réponse dans le cours")
    st.caption("Questions que le tuteur n'a pas retrouvées dans le support : passages peu clairs ou absents ?")
    if len(unans):
        st.dataframe(unans[["date", "question"]].tail(30).iloc[::-1], hide_index=True)
    else:
        st.info("Rien à signaler.")

    st.subheader("Énergie et stress moyens du groupe, par semaine")
    if len(debut):
        wk = debut.set_index("date")[["energie", "stress"]].resample("W").mean()
        st.line_chart(wk)

    st.subheader("Export des données (CSV)")
    for name, label in [("sessions", "Séances"), ("quiz", "Quiz"), ("fiches", "Fiches"), ("plans", "Plans"), ("pauses", "Pauses")]:
        st.download_button(f"Télécharger : {label}", read(name).to_csv(index=False).encode("utf-8"), f"{name}.csv", "text/csv")


# =====================================================================
# ACCUEIL ET NAVIGATION
# =====================================================================
if st.query_params.get("formateur"):
    page_formateur()
    st.stop()

if "code" not in s:
    st.title("🚀 Mon espace d'apprentissage")
    st.info(
        "Cet espace t'aide à progresser, à t'organiser et à garder de l'énergie. "
        "Le tuteur IA n'est pas un soutien psychologique. Ne saisis jamais ton nom, ton employeur "
        "ou d'autres données personnelles. Tes échanges sont enregistrés de façon pseudonymisée "
        "dans le cadre d'une étude."
    )
    code = st.text_input("Ton code apprenant (donné par ton formateur)").strip()
    ok = st.checkbox("J'ai lu ces informations et j'accepte de participer à l'étude")
    if st.button("Entrer") and code and ok:
        s.code = code
        st.rerun()
    st.stop()

PAGES = {
    "🏠 Mon tableau de bord": page_dashboard,
    "💬 Mon tuteur": page_tuteur,
    "🧠 Je me teste": page_quiz,
    "📅 Mon plan": page_plan,
    "🌿 Ma pause": page_pause,
    "🤝 Besoin d'un humain ?": page_humain,
}
with st.sidebar:
    st.write(f"Connecté : **{s.code}**")
    choice = st.radio("Menu", list(PAGES), label_visibility="collapsed")
PAGES[choice]()
