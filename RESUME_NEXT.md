# RESUME NEXT — riprendere il benchmark dopo il riavvio

**Stato al 2026-06-17:** benchmark **non completato**. La key di Kekko è **valida** ma
il suo progetto Google ha quote free-tier troppo basse. Branch: `fix/honesty-pass` (PR #1).

## Quote misurate su questa key (entrambe bloccanti)

| Limite | Valore | Modello | Fonte |
|--------|--------|---------|-------|
| **Per minuto** | **5 req/min** | gemini-2.5-flash | 429 `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, retryDelay 41s |
| **Per giorno** | ~20 req/giorno | gemini-2.5-flash-lite | 429 `…PerDay…`, quotaValue 20 |
| Per giorno | **0** | gemini-2.0-flash | quotaValue 0 |

Il benchmark serve **~480 call** in pochi minuti → impossibile con 5/min + ~20/giorno.

## Le due strade per sbloccare (scegline una)

### A) Key migliore (gratis, consigliata)
1. https://aistudio.google.com/apikey → **Create API key** in un **progetto nuovo**.
2. Deve iniziare con `AIza…`. Il free tier standard dà ~10–15 RPM e 200–1500/giorno.
3. ```powershell
   $env:GEMINI_API_KEY="AIza..."; $env:CAMBRIAN_BACKEND="gemini"
   $env:CAMBRIAN_GEMINI_MODEL="gemini-2.5-flash"
   cd C:\Users\vault01\cambrian
   python benchmarks/humaneval_real.py --problems 5 --budget 24 --population 6 --date 2026_06_17
   ```

### B) Restare su questa key ma rispettare i 5 RPM (lento ma fa-zero-costo)
Il backend ora ritenta fino a 8 volte (backoff ~2^8s) per cavalcare la finestra da 1 min.
Ma a 5 RPM un run da ~480 call richiede **~100 minuti** e potrebbe sbattere sul cap
giornaliero. Per un **pilot reale piccolo** che probabilmente passa:
```powershell
$env:GEMINI_API_KEY="<LA_KEY_CHE_HAI_FORNITO>"   # la stessa AQ.… di prima
$env:CAMBRIAN_BACKEND="gemini"; $env:CAMBRIAN_GEMINI_MODEL="gemini-2.5-flash"
$env:CAMBRIAN_GEMINI_RETRIES="10"
cd C:\Users\vault01\cambrian
# ~13 call totali, ~3-4 min con i ritardi del rate-limit. NON è un verdetto (n=1).
python benchmarks/humaneval_real.py --problems 1 --budget 3 --population 3 --date 2026_06_17
```
Aspetta che la quota **giornaliera** sia fresca (reset a mezzanotte Pacific Time) prima di B.

## Cosa è già pronto (non rifare)
- `benchmarks/humaneval_real.py` — harness completo, ora con: abort onesto su quota
  esaurita (niente più zeri falsi), backoff lungo per i 5 RPM, modello via env.
- Output di un tentativo: `benchmarks/results/humaneval_2026_06_17_ABORTED.json`
  (status ABORTED — prova documentale del blocco quota, non un verdetto).
- Phase A (stub rimosso, badge onesti, operatori connessi) — già committata sul branch.

## Verdetto attuale: 🟡 POLISH + RIPROVA
Invariato. Tutto pronto. L'unico gate è una key con quota adeguata (strada A, ~3 min).
Appena il benchmark gira: GREEN >15pp → PROMUOVI · YELLOW 5–15pp → polish · RED <5pp → ARCHIVIA.
