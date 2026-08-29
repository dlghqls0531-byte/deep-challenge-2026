
import time, gc, numpy as np, torch, mathx

def build_prompt(tok, q):
    return tok.apply_chat_template(
        [{"role": "system", "content": mathx.PROMPT_SYSTEM},
         {"role": "user",   "content": mathx.PROMPT_USER.format(q=q)}],
        tokenize=False, add_generation_prompt=True)

def _make_batches(L, order, batch_size, token_budget):
    """(배치 크기 × 배치 내 최대길이) <= token_budget 이 되도록 묶는다"""
    batches, cur = [], []
    for i in order:
        t = cur + [i]
        if cur and (len(t) * int(L[t].max()) > token_budget or len(t) > batch_size):
            batches.append(cur); cur = [i]
        else:
            cur = t
    if cur: batches.append(cur)
    return batches

@torch.inference_mode()
def _gen_one(model, tok, texts, idxs, mnt, kw, max_input, depth=0):
    """OOM 시 배치를 절반으로 쪼개 재시도. 반환: {global_idx: (text, ntok)}"""
    out = {}
    try:
        enc = tok([texts[i] for i in idxs], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_input).to(model.device)
        g = model.generate(**enc, max_new_tokens=mnt, **kw)
        new = g[:, enc["input_ids"].shape[1]:]
        for row, i in zip(new, idxs):
            ids = row.tolist()
            while ids and ids[-1] in (tok.pad_token_id, tok.eos_token_id): ids.pop()
            out[i] = (tok.decode(ids, skip_special_tokens=True), len(ids))
        del enc, g, new
        return out
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        if len(idxs) == 1:
            print(f"      [OOM] 단일 샘플 실패 idx={idxs[0]} — 빈 출력 처리", flush=True)
            return {idxs[0]: ("", 0)}
        h = len(idxs)//2
        if depth == 0:
            print(f"      [OOM] 배치 {len(idxs)} → {h}+{len(idxs)-h} 분할 재시도", flush=True)
        out.update(_gen_one(model, tok, texts, idxs[:h], mnt, kw, max_input, depth+1))
        out.update(_gen_one(model, tok, texts, idxs[h:], mnt, kw, max_input, depth+1))
        return out

@torch.inference_mode()
def generate_rounds(model, tok, questions, rounds=(512, 512), batch_size=64,
                    do_sample=False, temperature=0.8, top_p=0.95, seed=None,
                    sort_by_len=True, max_input=1792, token_budget=45000,
                    tag="", verbose=True):
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    n = len(questions)
    base = [build_prompt(tok, q) for q in questions]
    partial = [""]*n; done=[False]*n; ntok=[0]*n
    t0 = time.time(); n_oom = 0
    for ri, mnt in enumerate(rounds):
        idx = [i for i in range(n) if not done[i]]
        if not idx: break
        texts = {i: base[i] + partial[i] for i in idx}
        L = np.zeros(n, dtype=np.int64)
        for i in idx:
            L[i] = min(len(tok(texts[i], add_special_tokens=False)["input_ids"]), max_input)
        order = sorted(idx, key=lambda i: L[i]) if sort_by_len else idx
        batches = _make_batches(L, order, batch_size, token_budget)
        kw = dict(do_sample=do_sample, pad_token_id=tok.pad_token_id,
                  eos_token_id=tok.eos_token_id)
        if do_sample: kw.update(temperature=temperature, top_p=top_p)
        for bi, b in enumerate(batches):
            res = _gen_one(model, tok, texts, b, mnt, kw, max_input)
            for i, (seg, k) in res.items():
                partial[i] += seg; ntok[i] += k
                if k < mnt or mathx.end_kind(partial[i]) == "clean_final":
                    done[i] = True
            torch.cuda.empty_cache()
        if verbose:
            print(f"    [{tag}] round{ri+1} mnt={mnt} 처리={len(idx):5d} "
                  f"배치={len(batches):4d} 누적완료={sum(done):5d}/{n} "
                  f"{time.time()-t0:7.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)
    el = time.time() - t0
    return dict(texts=partial, finished=done, ntok=ntok, elapsed=el,
                useful_tok=int(sum(ntok)), tok_per_sec=sum(ntok)/max(el,1e-9))

def score(texts, gold):
    pred = [mathx.extract(t, "lenient")[0] for t in texts]
    ek   = [mathx.end_kind(t) for t in texts]
    ok   = np.array([(p is not None) and (p == g) for p, g in zip(pred, gold)])
    return ok, pred, ek
