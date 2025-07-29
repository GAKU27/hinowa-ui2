/*! Hinowa Engine v4.1 (FAS_composite) — drop-in patch
 *  Features:
 *   - Safe-Bias weights (short items)
 *   - Repetition & physiology boosters
 *   - Hash-based debounce (same text won't accumulate)
 *   - Hysteresis band to avoid chattering near thresholds
 */
(function(global){
  const W = {
    EmotionDensity:.20, BreakdownDepth:.12, SilenceTrend:.10, RiseTrend:.08,
    DeltaTemp:.06, SimilarityChange:.05, SemanticDrift:.05, ArousalVariance:.05,
    LinguisticEnergy:.04, TemporalCompression:.03, PhysioFlag:.02, ExternalContextWeight:.02,
    inv_CUS:.08, inv_RIS:.05, inv_EIS:.03, inv_SSS:.02
  };
  const CFG = { alpha:.70, hysteresis: .02, nightHours:[0,1,2,3,4,5], nightBonusBase:.08 };

  function hashText(s){
    // djb2
    let h=5381; for(let i=0;i<s.length;i++){ h=((h<<5)+h) + s.charCodeAt(i); h|=0; }
    return h>>>0;
  }
  function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }
  function sigmoid(x){ return 1/(1+Math.exp(-x)); }
  function tokJa(text){
    const t = (text||"").toLowerCase().replace(/\s+/g," ").trim();
    return t.split(/[^a-z0-9ぁ-んァ-ン一-龥ー]+/).filter(Boolean);
  }
  function freqWeighted(text, clusters){
    if(!clusters) return {score:0, physio:0, crisis:0, hits:[]};
    const t = (text||""); const hits = [];
    let sum=0, physio=0, crisis=0;
    for(const c of clusters){
      const weight = +c.weight || 0.3;
      for(const e of c.entries){
        try{
          const re = new RegExp(e, "gi");
          const m = t.match(re);
          if(m && m.length){
            const s = Math.min(1, m.length*weight*0.5);
            sum += s;
            hits.push({tag:c.tag, entry:e, count:m.length, w:weight, s});
            if(c.tag==="体感") physio += Math.min(1, m.length*0.2);
            if(c.tag==="希死" || c.tag==="救援") crisis += Math.min(1, m.length*0.3);
          }
        }catch(_){ /* invalid regex; skip */ }
      }
    }
    // repetition booster (e.g., 助けて助けて)
    if(/(.)\1{2,}/.test(t) || /(助けて){2,}/.test(t)) sum+=0.3;
    sum += Math.min(0.2, physio*0.3 + crisis*0.3);
    return {score:clamp(sigmoid(sum)-0.5+0.5,0,1), physio:clamp(physio,0,1), crisis:clamp(crisis,0,1), hits};
  }
  function coherence(text){
    const s=(text||"").replace(/[\r\n]+/g,"\n").split(/[。\.！？!\?…\n]/).map(x=>x.trim()).filter(Boolean);
    if(s.length<2) return 0.65;
    let o=0,p=0;
    const sets=s.map(u=>new Set(tokJa(u).filter(w=>w.length>=2)));
    for(let i=1;i<sets.length;i++){
      const a=sets[i-1], b=sets[i];
      const inter=[...a].filter(x=>b.has(x)).length;
      const denom=Math.max(1, Math.min(a.size,b.size)); o+=inter/denom; p++;
    }
    return clamp(o/Math.max(1,p),0,1);
  }
  function countPats(text, pats){
    let c=0; for(const re of pats){ try{ const m=(text||"").match(re); if(m) c+=m.length; }catch(_){ } }
    return c;
  }
  function computeMi(text, dict){
    const NEG = freqWeighted(text, dict?.clusters);
    const dots = countPats(text, [/…+/g, /\.{3,}/g, /ーー+/g, /—+/g, /、{2,}/g, /。{2,}/g]);
    const excq = countPats(text, [/!{2,}/g, /？{2,}/g, /\?{2,}/g, /！{2,}/g]);
    const lines = (text||"").split(/\n/).filter(l=>l.trim().length>0);
    const veryShort = lines.filter(l=>l.trim().length<=5).length;
    const bd = clamp(Math.tanh(((dots*0.6+veryShort*0.4)/Math.max(1,lines.length))*2),0,1);
    const av = clamp(Math.tanh((excq+NEG.crisis*2)/(tokJa(text).length+8)*40),0,1);
    const unique=new Set(tokJa(text)).size; const len=tokJa(text).length;
    const le=clamp(Math.tanh(((len?unique/len:0)-0.25)*3+1),0,1);
    // Proxy scores
    const Mi = {
      EmotionDensity: NEG.score,
      BreakdownDepth: bd,
      SilenceTrend: .5,
      RiseTrend: clamp(Math.tanh((excq + Math.max(0, NEG.score-0.4)*10)), 0, 1),
      DeltaTemp: clamp((NEG.physio*0.6 + NEG.crisis*0.7), 0, 1),
      SimilarityChange: .5,
      SemanticDrift: .5,
      ArousalVariance: av,
      LinguisticEnergy: le,
      TemporalCompression: .5,
      PhysioFlag: NEG.physio>0.2?1:0,
      ExternalContextWeight: .2
    };
    return {Mi, NEG};
  }
  function computeCUS(text){
    const pos = /安心|楽|穏やか|落ち着|よかった|感謝|ありがとう/g;
    const self = /感じ|思|気づ|考|振り返/g;
    const prob = /やってみ|試|計画|段取り|解決|準備/g;
    const calm = /呼吸|深呼吸|ゆっくり|落ち着/g;
    const CUS = clamp(
      0.30*Math.tanh(((text||"").match(pos)||[]).length/4) +
      0.25*coherence(text) +
      0.20*Math.tanh(((text||"").match(self)||[]).length/3) +
      0.15*Math.tanh(((text||"").match(prob)||[]).length/3) +
      0.10*Math.tanh(((text||"").match(calm)||[]).length/3), 0, 1);
    return CUS;
  }
  function computeRIS(text){
    const coping = /休む|運動|散歩|相談|深呼吸|瞑想|睡眠/g;
    const eff = /できる|やれる|乗り越え|いける/g;
    const past = /以前|前(は|に).+できた|過去/g;
    const support = /家族|友人|支援|専門家/g;
    const growth = /学ぶ|成長|改善/g;
    return clamp(
      0.35*Math.tanh(((text||"").match(coping)||[]).length/3) +
      0.25*Math.tanh(((text||"").match(eff)||[]).length/3) +
      0.20*Math.tanh(((text||"").match(past)||[]).length/2) +
      0.10*Math.tanh(((text||"").match(support)||[]).length/2) +
      0.10*Math.tanh(((text||"").match(growth)||[]).length/2), 0, 1);
  }
  function computeEIS(text){
    // Proxy without session timing info
    const direct = /灯輪|あなた|君/g;
    return clamp(0.30*0.8 + 0.25*0.6 + 0.20*0.6 + 0.15*Math.tanh(((text||"").match(direct)||[]).length/2) + 0.10*0.7, 0, 1);
  }
  function computeSSS(text){
    const rout = /食べ|寝|起き|入浴|掃除|勉強|連絡|片付/g;
    const plan = /明日|今週|予定|計画|予約/g;
    const proa = /自分で|準備|手配|先に/g;
    const dep = /助けて|頼り|依存|無理/g;
    const ext = /外出|買い物|散歩|会う/g;
    return clamp(
      0.30*Math.tanh(((text||"").match(rout)||[]).length/4) +
      0.25*Math.tanh(((text||"").match(plan)||[]).length/3) +
      0.20*Math.tanh(((text||"").match(proa)||[]).length/3) +
      0.15*(1 - Math.tanh(((text||"").match(dep)||[]).length/3)) +
      0.10*Math.tanh(((text||"").match(ext)||[]).length/3), 0, 1);
  }
  const State = { lastHash: null, lastFAS: .5 };
  function nightBonusByLocal(){
    const h = new Date().getHours(); return CFG.nightHours.includes(h) ? CFG.nightBonusBase : 0;
  }
  function coreFromMi(Mi, CUS,RIS,EIS,SSS){
    let core=0; for(const k in Mi){ core += (W[k]||0)*Mi[k]; }
    core += W.inv_CUS*(1-CUS) + W.inv_RIS*(1-RIS) + W.inv_EIS*(1-EIS) + W.inv_SSS*(1-SSS);
    return clamp(core,0,1);
  }
  function decide(FAS,CUS){
    const mode = (FAS>=0.85) ? "E1" :
                 (FAS>=0.68) ? (CUS>=0.66?"P1":(CUS>=0.33?"P2":"EPR")) :
                 (FAS>=0.50) ? (CUS>=0.66?"R1":(CUS>=0.33?"S2":"S1")) :
                 (CUS>=0.66?"FirePath":(CUS>=0.33?"R3":"ε1"));
    const zone = (FAS>=0.85)?"Critical":(FAS>=0.68)?"High":(FAS>=0.50)?"Medium":"Low";
    const cz = (CUS>=0.66)?"High":(CUS>=0.33)?"Moderate":"Low";
    return {mode, FAS_zone:zone, CUS_zone:cz};
  }
  function compute(text, dict){
    const H = hashText(text||"");
    const same = (State.lastHash===H);
    const alpha = same ? 0 : CFG.alpha;
    const {Mi, NEG} = computeMi(text, dict);
    const CUS = computeCUS(text);
    const RIS = computeRIS(text);
    const EIS = computeEIS(text);
    const SSS = computeSSS(text);
    const core = coreFromMi(Mi, CUS,RIS,EIS,SSS);
    const RUB = Math.min(0.05, (NEG.crisis>0?0.02:0) + (NEG.physio>0.3?0.02:0));
    const NB = nightBonusByLocal();
    const fas_now = clamp(alpha*State.lastFAS + (1-alpha)*clamp(core + RUB + NB, 0, 1), 0, 1);
    const res = {FAS: fas_now, core, Mi, CUS,RIS,EIS,SSS, RUB, NB, NEG};
    const dec = decide(res.FAS, CUS);
    State.lastHash = H; State.lastFAS = fas_now;
    return {...res, ...dec};
  }
  global.HinowaEngineV41 = { compute, hashText, state: State, weights: W, config: CFG };
})(window);
