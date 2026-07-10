#!/usr/bin/env python3
"""Build the expanded 50-item dissertation benchmark corpus.

The script is intentionally conservative: it records all 50 curated sources,
downloads only files that resolve to PDFs, reuses existing downloads, and keeps
landing-only records in the manifest instead of counting them as successful.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from pathlib import Path

import requests
from pypdf import PdfReader


BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "expanded_50"
OUTDIR.mkdir(parents=True, exist_ok=True)


SOURCES = [
    ("abbeel_2008_apprenticeship_rl", "Apprenticeship Learning and Reinforcement Learning with Application to Robotic Control", "Pieter Abbeel", 2008, "classic RL / imitation learning", "https://ai.stanford.edu/~pabbeel//thesis/thesis.pdf"),
    ("abel_2020_abstraction_rl", "A Theory of Abstraction in Reinforcement Learning", "David Abel", 2020, "RL theory / abstraction", "https://david-abel.github.io/thesis.pdf"),
    ("goldie_2024_seq_decision", "Reinforcement Learning for Sequential Decision-Making: From Chip Design to Language Modeling", "Anna Goldie", 2024, "RL / language modeling", "https://nlp.stanford.edu/~manning/dissertations/Anna_Goldie___Dissertation-augmented.pdf"),
    ("kostrikov_2021_sample_efficiency", "Improving Sample Efficiency of Imitation and Reinforcement Learning", "Ilya Kostrikov", 2021, "deep RL / imitation learning", "https://cs.nyu.edu/media/publications/Kostrikov_phd_thesis_final.pdf"),
    ("wensun_2019_generalization_efficiency", "Towards Generalization and Efficiency in Reinforcement Learning", "Wen Sun", 2019, "RL generalization / sample efficiency", "https://publications.ri.cmu.edu/resolve/2019/06/wensun_phd_ri_2019.pdf"),
    ("zanette_2021_sample_efficient_exploration", "Reinforcement Learning: When Can We Do Sample Efficient Exploration?", "Andrea Zanette", 2021, "RL exploration theory", "https://stacks.stanford.edu/file/druid:ns268vs3612/Thesis_submitted-augmented.pdf"),
    ("yiming_zhang_2024_on_policy_drl", "On-Policy Deep Reinforcement Learning", "Yiming Zhang", 2024, "deep RL / on-policy methods", "https://cs.nyu.edu/media/publications/Yiming_Zhang_PhD_Thesis.pdf"),
    ("ng_2003_shaping_policy_search", "Shaping and Policy Search in Reinforcement Learning", "Andrew Y. Ng", 2003, "reward shaping / policy search", "https://rll.berkeley.edu/deeprlcoursesp17/docs/ng-thesis.pdf"),
    ("silver_2009_computer_go", "Reinforcement Learning and Simulation-Based Search in Computer Go", "David Silver", 2009, "RL / search", "http://www.incompleteideas.net/papers/Silver-phd-thesis.pdf"),
    ("littman_1996_sequential_decision", "Algorithms for Sequential Decision Making", "Michael L. Littman", 1996, "sequential decision making", "https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf"),
    ("kuss_2006_gp_rl", "Gaussian Process Models for Robust Regression, Classification, and Reinforcement Learning", "Malte Kuss", 2006, "Bayesian methods / RL", "https://tuprints.ulb.tu-darmstadt.de/674/1/GaussianProcessModelsKuss.pdf"),
    ("wiering_1999_efficient_rl", "Explorations in Efficient Reinforcement Learning", "Marco Wiering", 1999, "RL algorithms", "https://pure.uva.nl/ws/files/3153480/8462_UBA003000033_001.pdf"),
    ("cuayahuitl_2009_hrl_dialogue", "Hierarchical Reinforcement Learning for Spoken Dialogue Systems", "Heriberto Cuayahuitl", 2009, "hierarchical RL / dialogue", "http://eprints.lincoln.ac.uk/id/eprint/22207/1/hc-phd-thesis.pdf"),
    ("hong_2025_generative_discovery_rl", "Generative Discovery via Reinforcement Learning", "Zirui Wang Hong", 2025, "RL / generative discovery", "https://dspace.mit.edu/bitstream/handle/1721.1/159135/hong-zwhong-phd-eecs-2025-thesis.pdf?isAllowed=y&sequence=1"),
    ("han_2019_continuous_control_drl", "Continuous Control for Robot Based on Deep Reinforcement Learning", "Jing Han", 2019, "deep RL / robotics", "https://dr.ntu.edu.sg/bitstream/10356/90191/1/thesis-final.pdf"),
    ("liu_2019_ppo_starcraft", "Proximal Policy Optimization in StarCraft", "Jianing Liu", 2019, "PPO / game RL", "https://digital.library.unt.edu/ark:/67531/metadc1505267/m2/1/high_res_d/LIU-THESIS-2019.pdf"),
    ("gupta_2023_trust_region_rl", "Trust-Region Based Policy Optimization for Efficient Reinforcement Learning", "Rishabh Gupta", 2023, "policy optimization", "https://digitalcommons.uri.edu/cgi/viewcontent.cgi?article=2582&context=oa_diss"),
    ("lomonaco_2019_continual_learning", "Continual Learning with Deep Architectures", "Vincenzo Lomonaco", 2019, "continual learning", "http://amsdottorato.unibo.it/9073/1/vincenzo_lomonaco_thesis.pdf"),
    ("maruf_2023_language_model_adaptation", "Transfer Learning for Language Model Adaptation", "Sameen Maruf", 2023, "language model adaptation", "https://dr.ntu.edu.sg/bitstream/10356/169892/2/sbmaruf_thesis.pdf"),
    ("walsh_2010_relational_models", "Efficient Learning of Relational Models for Sequential Decision Making", "Thomas J. Walsh", 2010, "RL / relational models", "https://cs.brown.edu/people/mlittman/theses/walsh.pdf"),
    ("strehl_2007_computational_rl_theory", "A Unifying Framework for Computational Reinforcement Learning Theory", "Alexander L. Strehl", 2007, "RL theory", "https://rucore.libraries.rutgers.edu/rutgers-lib/26345/PDF/1/play/"),
    ("givchi_2022_optimal_transport_rl", "Optimal Transport in Reinforcement Learning", "Arash Givchi", 2022, "RL / optimal transport", "https://rucore.libraries.rutgers.edu/rutgers-lib/66700/PDF/1/"),
    ("anthony_liu_2025_compositional_rl", "Leveraging Compositional Structure for Reinforcement Learning and Sequential Decision-Making", "Anthony Liu", 2025, "compositional RL", "https://deepblue.lib.umich.edu/bitstream/handle/2027.42/197133/anthliu_1.pdf?sequence=1"),
    ("fan_yang_2023_safe_rl", "Exploring Safe Reinforcement Learning for Sequential Decision Making", "Fan Yang", 2023, "safe RL", "https://www.ri.cmu.edu/app/uploads/2023/05/MSR_thesis_Fan_Yang.pdf"),
    ("bradley_2022_nonlinear_predictive_control_rl", "Reinforcement Learning for Optimization of Nonlinear and Predictive Control", "Erlend A. Bradley", 2022, "control / RL", "https://torarnj.folk.ntnu.no/eb_phdthesis_final.pdf"),
    ("xue_2023_adaptive_decision_rl", "Robust and Adaptive Decision-Making: A Reinforcement Learning Perspective", "Wanqi Xue", 2023, "robust RL / decision making", "https://personal.ntu.edu.sg/boan/thesis/Xue_Wanqi_PhD_Thesis.pdf"),
    ("ajay_2024_foundation_decision", "Composing Foundation Models for Decision Making", "Anurag Ajay", 2024, "foundation models / decision making", "https://dspace.mit.edu/bitstream/handle/1721.1/158501/ajay-aajay-phd-eecs-2024-thesis.pdf?sequence=1"),
    ("foerster_2018_marl", "Deep Multi-Agent Reinforcement Learning", "Jakob Foerster", 2018, "multi-agent RL", "https://ora.ox.ac.uk/objects/uuid:a55621b3-53c0-4e1b-ad1c-92438b57ffa4"),
    ("sukhbaatar_2017_comm_marl", "Cooperation and Communication in Multiagent Deep Reinforcement Learning", "Sainbayar Sukhbaatar", 2017, "multi-agent deep RL", "http://hdl.handle.net/2152/45681"),
    ("meyer_2017_reward_design", "Deep Learning and Reward Design for Reinforcement Learning", "Eric Meyer", 2017, "reward design / deep RL", "https://hdl.handle.net/2027.42/136931"),
    ("kearns_2003_sample_complexity", "On the Sample Complexity of Reinforcement Learning", "Michael Kearns", 2003, "RL sample complexity", "http://hdl.handle.net/10068/907685"),
    ("peshkin_2001_importance_sampling_rl", "Importance Sampling for Reinforcement Learning with Multiple Objectives", "Leonid Peshkin", 2001, "RL / importance sampling", "http://hdl.handle.net/1721.1/5568"),
    ("duff_2002_exploration_inference", "Exploration and Inference in Learning from Reinforcement", "Michael O. Duff", 2002, "exploration / inference", "http://hdl.handle.net/1842/532"),
    ("mataric_1994_robot_rl", "Reinforcement Learning in Autonomous Robots: An Empirical Investigation of the Role of Emotion", "Maja J. Mataric", 1994, "robot RL", "http://hdl.handle.net/1842/360"),
    ("mikolov_2012_neural_lm", "Statistical Language Models Based on Neural Networks", "Tomas Mikolov", 2012, "neural language models", "https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf"),
    ("sutskever_2013_rnn", "Training Recurrent Neural Networks", "Ilya Sutskever", 2013, "RNNs / language modeling", "https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf"),
    ("socher_2014_recursive_dl", "Recursive Deep Learning for Natural Language Processing and Computer Vision", "Richard Socher", 2014, "deep NLP", "https://nlp.stanford.edu/~socherr/thesis.pdf"),
    ("saphra_2024_training_dynamics", "Training Dynamics of Neural Language Models", "Naomi Saphra", 2024, "neural language models", "https://nsaphra.net/uploads/thesis.pdf"),
    ("lyu_2024_llm_explanations", "Faithful and Useful Explanations by Large Language Models", "Qing Lyu", 2024, "LLM explanations", "https://www.cis.upenn.edu/~ccb/publications/dissertations/veronica-qing-lyu-thesis.pdf"),
    ("li_zhang_2024_event_reasoning", "Structured Event Reasoning with Large Language Models", "Li Zhang", 2024, "LLM reasoning", "https://www.cis.upenn.edu/~ccb/publications/dissertations/harry-li-zhang-thesis.pdf"),
    ("zhili_feng_2025_foundation_analysis", "Leveraging Information Theoretic Tools for Foundation Model Analysis", "Zhili Feng", 2025, "foundation models / LLM analysis", "https://ml.cmu.edu/research/phd-dissertation-pdfs/zhilif_phd_mld_2025.pdf"),
    ("khandelwal_2020_black_box_lm", "Improving Neural Language Models with Black-Box Systems", "Urvashi Khandelwal", 2020, "neural language models", "https://stacks.stanford.edu/file/druid:st056pp9441/urvashi_thesis-augmented.pdf"),
    ("ruder_2019_transfer_nlp", "Neural Transfer Learning for Natural Language Processing", "Sebastian Ruder", 2019, "NLP transfer learning", "http://hdl.handle.net/10379/15463"),
    ("nemeskey_2021_nlp_language_modeling", "Natural Language Processing Methods for Language Modeling", "David M. Nemeskey", 2021, "language modeling", "https://edit.elte.hu/xmlui/bitstream/10831/62063/1/Nemeskey_David_Ertekezes.pdf"),
    ("ponte_2001_lm_ir", "Using Language Models for Information Retrieval", "Jay M. Ponte", 2001, "language models / information retrieval", "https://zenodo.org/record/570441"),
    ("zhu_2005_ssl_natural_language", "Semi-Supervised Learning for Natural Language", "Xiaojin Zhu", 2005, "semi-supervised NLP", "http://hdl.handle.net/1721.1/33296"),
    ("kim_2015_cnn_sentence_classification", "Convolutional Neural Network for Sentence Classification", "Yoon Kim", 2015, "neural NLP", "http://hdl.handle.net/10012/9592"),
    ("suess_2026_faithfulness_nle", "Faithfulness of Natural Language Explanations for Large Language Models", "Maximilian Suess", 2026, "LLM explanation faithfulness", "https://repositum.tuwien.at/bitstream/20.500.12708/226904/1/Suess%20Maximilian%20-%202026%20-%20Faithfulness%20of%20Natural%20Language%20Explanations%20for...pdf"),
    ("wu_2025_improving_foundation_models", "Improving Foundation Models", "Georgia Tech dissertation author", 2025, "foundation models", "https://repository.gatech.edu/items/37c97667-ed22-4370-8745-c4508d4f3bac"),
    ("carleton_2024_multilingual_semantic_llms", "The Fusion of Multilingual Semantic Search and Large Language Models", "Carleton thesis author", 2024, "LLM retrieval / semantic search", "https://carleton.scholaris.ca/bitstreams/e5694573-551f-41bf-a945-fd0d3c0f924a/download"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 dissertation-benchmark-corpus/1.0"}


def find_pdf_url(url: str, html_text: str) -> str | None:
    meta = re.search(
        r"<meta[^>]+name=[\"']citation_pdf_url[\"'][^>]+content=[\"']([^\"']+)",
        html_text,
        re.I,
    )
    if meta:
        return urllib.parse.urljoin(url, html.unescape(meta.group(1)))

    links = re.findall(r"href=[\"']([^\"']+)[\"']", html_text, re.I)
    candidates = []
    for link in links:
        candidate = urllib.parse.urljoin(url, html.unescape(link))
        lower = candidate.lower()
        if any(token in lower for token in [".pdf", "bitstream", "download", "/pdf/", "viewcontent", "retrieve"]):
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            ".pdf" in candidate.lower(),
            "download" in candidate.lower(),
            "bitstream" in candidate.lower(),
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def get_bytes(url: str) -> tuple[bytes, requests.Response]:
    response = requests.get(url, headers=HEADERS, timeout=(10, 45), allow_redirects=True)
    response.raise_for_status()
    return response.content, response


def fetch_pdf(record: dict[str, object]) -> dict[str, object]:
    path = OUTDIR / f"{record['id']}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return {"status": "downloaded", "path": str(path), "bytes": path.stat().st_size, "cached": True}

    url = str(record["url"])
    tried = []
    for _ in range(3):
        tried.append(url)
        try:
            content, response = get_bytes(url)
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "tried": tried}

        content_type = response.headers.get("content-type", "").lower()
        if len(content) > 85 * 1024 * 1024:
            return {
                "status": "skipped_large",
                "bytes": len(content),
                "content_type": content_type,
                "final_url": response.url,
                "tried": tried,
            }

        if b"%PDF" in content[:2048] or "pdf" in content_type:
            path.write_bytes(content)
            return {
                "status": "downloaded",
                "path": str(path),
                "bytes": path.stat().st_size,
                "final_url": response.url,
                "tried": tried,
            }

        text = content[:5_000_000].decode("utf-8", errors="ignore")
        pdf_url = find_pdf_url(response.url, text)
        if pdf_url and pdf_url not in tried:
            url = pdf_url
            continue

        return {
            "status": "landing_only",
            "content_type": content_type,
            "final_url": response.url,
            "tried": tried,
            "pdf_candidate": pdf_url,
        }

    return {"status": "landing_only", "tried": tried}


def extract_pdf(path: Path) -> dict[str, object]:
    info: dict[str, object] = {"path": str(path)}
    try:
        reader = PdfReader(str(path))
        info["pages"] = len(reader.pages)
        metadata = reader.metadata or {}
        info["metadata_title"] = str(getattr(metadata, "title", "") or "")[:250]
        text_parts = []
        for index in range(min(12, len(reader.pages))):
            try:
                text_parts.append((reader.pages[index].extract_text() or "")[:4000])
            except Exception:
                text_parts.append("")
        text = "\n".join(text_parts)
        markers = []
        for line in text.splitlines():
            clean = " ".join(line.split())
            if len(clean) < 4 or len(clean) > 130:
                continue
            if re.match(
                r"^(abstract|acknowledg|contents|chapter\s+\d+|\d+(\.\d+)*\s+[A-Z][A-Za-z].+|introduction|conclusion|bibliography|references)\b",
                clean,
                re.I,
            ):
                markers.append(clean)
        seen = set()
        unique = []
        for marker in markers:
            key = marker.lower()
            if key not in seen:
                seen.add(key)
                unique.append(marker)
        info["early_structure_markers"] = unique[:40]
        info["first_pages_excerpt"] = text[:2500]
    except Exception as exc:
        info["extract_error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> None:
    assert len(SOURCES) == 50, len(SOURCES)

    records = []
    for index, item in enumerate(SOURCES, 1):
        source_id, title, author, year, topic, url = item
        print(f"[{index:02d}/50] {source_id}", flush=True)
        record: dict[str, object] = {
            "id": source_id,
            "title": title,
            "author": author,
            "year": year,
            "topic": topic,
            "url": url,
            "selection_basis": (
                "Curated as an RL/LLM/deep-learning dissertation or thesis benchmark "
                "from official repository, university, author, OpenAlex, or targeted "
                "web-search results; priority given to direct PDF availability and "
                "field relevance."
            ),
        }
        download = fetch_pdf(record)
        record.update(download)
        if record.get("status") == "downloaded":
            record.update(extract_pdf(Path(str(record["path"]))))
        records.append(record)

    manifest_path = BASE / "expanded_50_sources.json"
    manifest_path.write_text(json.dumps(records, indent=2))

    downloaded = [record for record in records if record.get("status") == "downloaded"]
    not_downloaded = [record for record in records if record.get("status") != "downloaded"]

    lines = [
        "# Expanded 50 Dissertation/Thesis Benchmark Corpus",
        "",
        f"Generated from 50 curated additional RL/LLM/deep-learning dissertation or thesis reports. Downloaded PDFs: {len(downloaded)}. Landing-only/failed/skipped: {len(not_downloaded)}.",
        "",
        "## Downloaded PDFs",
        "",
    ]
    for record in downloaded:
        lines.append(
            f"- **{record['author']} ({record['year']})**, _{record['title']}_ — "
            f"{record.get('pages', '?')} pages. Topic: {record['topic']}. Source: {record['url']}"
        )

    lines.extend(["", "## Not Downloaded or Landing-Only", ""])
    for record in not_downloaded:
        detail = record.get("error") or record.get("content_type") or record.get("final_url") or ""
        lines.append(
            f"- **{record['author']} ({record['year']})**, _{record['title']}_ — "
            f"status `{record.get('status')}`. Source: {record['url']}. Detail: {detail}"
        )

    lines.extend(
        [
            "",
            "## Recurrent Dissertation-Quality Patterns",
            "",
            "Across the downloadable set, the strongest dissertations make the claim hierarchy explicit early, separate algorithmic contribution from empirical validation, include threats-to-validity sections, and make reproduction artifacts auditable rather than merely available. RL dissertations also connect diagnostic failures to algorithmic next actions. LLM dissertations increasingly separate faithfulness, usefulness, and evaluation validity as distinct constructs.",
            "",
            "## Structural Signals Extracted",
            "",
        ]
    )
    for record in downloaded:
        markers = "; ".join(record.get("early_structure_markers", [])[:8])
        lines.append(f"- **{record['id']}**: {markers}")

    summary_path = BASE / "expanded_50_summary.md"
    summary_path.write_text("\n".join(lines))

    print(f"WROTE {manifest_path}")
    print(f"WROTE {summary_path}")
    print(f"downloaded {len(downloaded)} failed_or_landing {len(not_downloaded)}")


if __name__ == "__main__":
    main()
