import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from tqdm import tqdm
from llama_cpp import Llama

SCHEMA_KEYS = ["Age", "Symptom", "Medicine", "Health_Condition", "Specialist", "Medical_Procedure"]


# ======================
# SYSTEM PROMPTS (variants)
# ======================


SYSTEM_PROMPT_BASE_EN = ( # this system promt will use as zero shot (without examples)
    "You are a question-answering assistant that performs medical Named Entity Recognition (NER).\n"
    "For each question, identify ONLY the entities that are explicitly present in the provided text and answer STRICTLY in JSON.\n"
    "Your answer MUST be a single JSON object with EXACTLY these six keys (even if empty), in this order:\n"
    "{\"Age\":[],\"Symptom\":[],\"Medicine\":[],\"Health_Condition\":[],\"Specialist\":[],"
    "\"Medical_Procedure\":[]}\n\n"
    "Rules:\n"
    "- Copy spans verbatim from the text (Bangla/English as they appear). No paraphrasing or hallucination.\n"
    "- Duration vs Age: if a number modifies time words (\"গত/পিছনের/ধরে\" + দিন/সপ্তাহ/মাস, or "
    "\"last/for/since\" + days/weeks/months), DO NOT label it as Age.\n"
    "- Negation: if a symptom is negated within ~5 tokens (\"না/নাই/নেই/করিনি/হয়নি/হয়ে নাই/হয় নি\"), "
    "DO NOT extract it.\n"
    "- Lab/test terms alone (e.g., Triglyceride, কোলেস্টেরল, HbA1c) are NOT symptoms unless the text explicitly "
    "states a complaint.\n"
    "- Prefer concise head+modifier spans; exclude extra function words/punctuation. Do not output standalone "
    "single letters (e.g., X/RT/S).\n"
    "- Only label Age when it is an age expression (e.g., \"৪০ বছর\", \"years old\", \"Age 27\"), not bare numbers.\n"
    "- Lists contain strings only; no duplicates; no commentary before or after the JSON.\n\n"
    "Label hints (recall boosters without adding false positives):\n"
    "- Symptom: Extract single-word symptoms (e.g., \"জ্বর\", \"কাশি\", \"বমি\") when a complaint verb appears nearby "
    "(\"আছে/হচ্ছে/লাগছে/অনুভব/ভুগছি/সমস্যা\").\n"
    "- Symptom: Also extract collocates exactly (\"মাথা ব্যথা\", \"গলা ব্যথা\", \"ঘন কফ\", \"বুকে ব্যথা\").\n"
    "- Health_Condition: Keep disease/diagnosis nouns (\"ডায়াবেটিস\", \"উচ্চ রক্তচাপ\", \"অ্যাজমা\", "
    "\"থাইরয়েড\", \"মাইগ্রেন\", \"গ্যাস্ট্রাইটিস\").\n"
    "- Specialist: Titles/specialities (\"ডাক্তার\", \"চিকিৎসক\", \"বিশেষজ্ঞ\", \"ইএনটি বিশেষজ্ঞ\", "
    "\"গ্যাস্ট্রোএন্টারোলজিস্ট\", \"নিউরোলজিস্ট\").\n"
    "- Medical_Procedure: Extract tests/imaging when performed/ordered (\"করা হয়েছে/করাতে বলেছেন\", \"done/ordered\").\n\n"
    "You will be asked the question:\n"
    "\"Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\"\n\n"
    "Examples:\n"
)

SYSTEM_PROMPT_BASE_BN = ( # এই system prompt টি zero-shot হিসেবে ব্যবহার হবে (উদাহরণ ছাড়া)
    "তুমি একজন প্রশ্ন-উত্তর সহকারী, যে মেডিক্যাল Named Entity Recognition (NER) করে।\n"
    "প্রতিটি প্রশ্নের জন্য, প্রদত্ত টেক্সটে স্পষ্টভাবে উপস্থিত এমন entity-গুলোই ONLY শনাক্ত করবে এবং উত্তর STRICTLY JSON-এ দেবে।\n"
    "তোমার উত্তর MUST একটি single JSON object হবে, যেখানে EXACTLY এই ছয়টি key থাকবে (খালি হলেও), এই order-এ:\n"
    "{\"Age\":[],\"Symptom\":[],\"Medicine\":[],\"Health_Condition\":[],\"Specialist\":[],"
    "\"Medical_Procedure\":[]}\n\n"
    "Rules:\n"
    "- টেক্সট থেকে span verbatim কপি করো (Bangla/English যেভাবে আছে সেভাবেই)। কোন paraphrasing বা hallucination করবে না।\n"
    "- Duration vs Age: যদি কোন সংখ্যা সময়ের শব্দকে modify করে (\"গত/পিছনের/ধরে\" + দিন/সপ্তাহ/মাস, অথবা "
    "\"last/for/since\" + days/weeks/months), তাহলে সেটিকে Age হিসেবে label করবে না।\n"
    "- Negation: যদি কোন symptom ~5 টোকেনের মধ্যে negated থাকে (\"না/নাই/নেই/করিনি/হয়নি/হয়ে নাই/হয় নি\"), "
    "তাহলে সেটি extract করবে না।\n"
    "- Lab/test term একা থাকলে (যেমন Triglyceride, কোলেস্টেরল, HbA1c) সেগুলো Symptom নয়, যদি না টেক্সটে স্পষ্টভাবে "
    "কোন অভিযোগ/সমস্যা হিসেবে বলা থাকে।\n"
    "- সংক্ষিপ্ত head+modifier span প্রাধান্য দাও; অতিরিক্ত function word/punctuation বাদ দাও। standalone "
    "একটি অক্ষর (যেমন X/RT/S) আউটপুট দেবে না।\n"
    "- Age শুধুমাত্র তখনই label করবে যখন সেটি age expression (যেমন \"৪০ বছর\", \"years old\", \"Age 27\")—bare number নয়।\n"
    "- Lists-এ শুধুই string থাকবে; duplicates থাকবে না; JSON-এর আগে বা পরে কোন commentary থাকবে না।\n\n"
    "Label hints (false positive না বাড়িয়ে recall বাড়ানোর জন্য):\n"
    "- Symptom: complaint verb কাছাকাছি থাকলে single-word symptom extract করো (যেমন \"জ্বর\", \"কাশি\", \"বমি\") "
    "(\"আছে/হচ্ছে/লাগছে/অনুভব/ভুগছি/সমস্যা\").\n"
    "- Symptom: collocate-ও ঠিক 그대로 extract করো (\"মাথা ব্যথা\", \"গলা ব্যথা\", \"ঘন কফ\", \"বুকে ব্যথা\").\n"
    "- Health_Condition: disease/diagnosis noun রাখো (\"ডায়াবেটিস\", \"উচ্চ রক্তচাপ\", \"অ্যাজমা\", "
    "\"থাইরয়েড\", \"মাইগ্রেন\", \"গ্যাস্ট্রাইটিস\").\n"
    "- Specialist: title/speciality (\"ডাক্তার\", \"চিকিৎসক\", \"বিশেষজ্ঞ\", \"ইএনটি বিশেষজ্ঞ\", "
    "\"গ্যাস্ট্রোএন্টারোলজিস্ট\", \"নিউরোলজিস্ট\").\n"
    "- Medical_Procedure: test/imaging extract করো যখন performed/ordered থাকে (\"করা হয়েছে/করাতে বলেছেন\", \"done/ordered\").\n\n"
    "তোমাকে এই প্রশ্নটি করা হবে:\n"
    "\"এই টেক্সটে কোন কোন entity (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) আছে?\"\n\n"
    "Examples:\n"
)



# Plain instruction-style English (non-question style, still JSON)
SYSTEM_PROMPT_BASE_BN_EN = (
    "You are a question-answering assistant that performs medical Named Entity Recognition (NER) for Bangla/ Bengali Language.\n"
    "For each question, identify ONLY the entities that are explicitly present in the provided text and answer STRICTLY in JSON.\n"
    "Your answer MUST be a single JSON object with EXACTLY these six keys (even if empty), in this order:\n"
    "{\"Age\":[],\"Symptom\":[],\"Medicine\":[],\"Health_Condition\":[],\"Specialist\":[],"
    "\"Medical_Procedure\":[]}\n\n"
    "Rules:\n"
    "1. Copy spans verbatim from the text (Bangla/English as they appear). No paraphrasing or hallucination.\n"
    "2. Duration vs Age: if a number modifies time words (\"গত/পিছনের/ধরে\" + দিন/সপ্তাহ/মাস, or "
    "\"last/for/since\" + days/weeks/months), DO NOT label it as Age.\n"
    "3. Negation: if a symptom is negated within ~5 tokens (\"না/নাই/নেই/করিনি/হয়নি/হয়ে নাই/হয় নি\"), "
    "DO NOT extract it.\n"
    "4. Lab/test terms alone (e.g., Triglyceride, কোলেস্টেরল, HbA1c) are NOT symptoms unless the text explicitly "
    "states a complaint.\n"
    "5. Prefer concise head+modifier spans; exclude extra function words/punctuation. Do not output standalone "
    "single letters (e.g., X/RT/S).\n"
    "6. Only label Age when it is an age expression (e.g., \"৪০ বছর\", \"years old\", \"Age 27\"), not bare numbers.\n"
    "7. Lists contain strings only; no duplicates; no commentary before or after the JSON.\n\n"
    "Label hints (recall boosters without adding false positives):\n"
    "8. Symptom: Extract single-word symptoms (e.g., \"জ্বর\", \"কাশি\", \"বমি\") when a complaint verb appears nearby "
    "(\"আছে/হচ্ছে/লাগছে/অনুভব/ভুগছি/সমস্যা\").\n"
    "9. Symptom: Also extract collocates exactly (\"মাথা ব্যথা\", \"গলা ব্যথা\", \"ঘন কফ\", \"বুকে ব্যথা\").\n"
    "10. Health_Condition: Keep disease/diagnosis nouns (\"ডায়াবেটিস\", \"উচ্চ রক্তচাপ\", \"অ্যাজমা\", "
    "\"থাইরয়েড\", \"মাইগ্রেন\", \"গ্যাস্ট্রাইটিস\").\n"
    "11. Specialist: Titles/specialities (\"ডাক্তার\", \"চিকিৎসক\", \"বিশেষজ্ঞ\", \"ইএনটি বিশেষজ্ঞ\", "
    "\"গ্যাস্ট্রোএন্টারোলজিস্ট\", \"নিউরোলজিস্ট\").\n"
    "12. Medical_Procedure: Extract tests/imaging when performed/ordered (\"করা হয়েছে/করাতে বলেছেন\", \"done/ordered\").\n\n"
    "You will be asked the question:\n"
    "\"Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\"\n\n"
    "Examples:\n"  
)




# ======================
# Base few-shot examples (English wrappers, Bangla/English texts)
# ======================

BASE_FEWSHOTS: List[str] = [
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত সপ্তাহ থেকে জ্বর আছে, রাতে কাশি বাড়ে, মাঝে মাঝে বমি হয়.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"জ্বর\",\"কাশি\",\"বমি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"বুকে ব্যথা হচ্ছে এবং শ্বাসকষ্ট আছে; খুব অস্বস্তি লাগছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\",\"অস্বস্তি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"রোগী অ্যাজমা ও গ্যাস্ট্রাইটিসের রোগী; গতকাল ধরা পড়েছে মাইগ্রেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"অ্যাজমা\",\"গ্যাস্ট্রাইটিস\",\"মাইগ্রেন\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"থাইরয়েড সমস্যা আছে; history of ডায়াবেটিস উল্লেখ আছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\",\"ডায়াবেটিস\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত চার মাস ধরে মাথা ব্যথা; কাশি না.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"মাথা ব্যথা\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Your serum Triglyceride is slightly raised; HbA1c 6.5%.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত তিন দিন ধরে কাশি ও জ্বর আছে। একজন মেডিসিন বিশেষজ্ঞ আমাকে সেফিক্সিম দিয়েছেন। এক্স-রে করা হয়নি.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"কাশি\",\"জ্বর\"], \"Medicine\":[\"সেফিক্সিম\"], "
        "\"Health_Condition\":[], \"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"রোগীর বয়স ৫৫ বছর। তিনি ডায়াবেটিস ও উচ্চ রক্তচাপের রোগী এবং মেটফরমিন ও লোসারটান খাচ্ছেন.\"\n"
        "Answer: {\"Age\":[\"৫৫ বছর\"], \"Symptom\":[], \"Medicine\":[\"মেটফরমিন\",\"লোসারটান\"], "
        "\"Health_Condition\":[\"ডায়াবেটিস\",\"উচ্চ রক্তচাপ\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"ইএনটি বিশেষজ্ঞ টিম্পানোমেট্রি করতে বলেছেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[\"ইএনটি বিশেষজ্ঞ\"], \"Medical_Procedure\":[\"টিম্পানোমেট্রি\"]}\n\n"
    ),
]

QA_FEWSHOTS_01: List[str] = [ 
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার পিত্ত থলিতে পাথর আছে। গত ২.৫ বছর যাবৎ এটা হয়েছে। যার আকার ৪ সেমি। এতদিন তীব্র কোন ব্যাথা ছিল না কিন্তু গত ৮ দিনের মধ্যে ৩ দিন পেটে তীব্র ব্যাথা হয়েছিল এবং পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়। পিত্ত থলিতে পাথর হলে কী পেট ব্যাথার সাথে সাথে পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়? এবং কেন এই চাপ দিলে ব্যাথা অনুভূত হয়?? অভিজ্ঞ ডাক্তারের পরামর্শ চাচ্ছি।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"ব্যাথা\",\"পেটে তীব্র ব্যাথা\",\"চাপ দিলে ব্যাথা\",\"পেট ব্যাথার\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পিত্ত থলিতে পাথর\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আস্সালামুয়ালাইকুম, আমার বয়স 22 বছর. গত দেড় মাস যাবত আমার পালস অনেক বেশি থাকে ( বেশি সময় 100 এর উপরে ) বিশেষ করে খাবার খাওয়ার পর. সাথে দুর্বলতা থাকায় এক মাস আগে আমি একজন মেডিসিন বিশেষজ্ঞ কে দেখাই. তিনি ECG এবং ECHO করাতে বলেন কিন্তু রিপোর্টে তেমন অস্বাভাবিক কিছু না থাকায় তিনি আমাকে PROPRANOLOL HCL 10 mg ৩০ দিন এবং ALPRAZOLAM 0. 25 mg ১০ দিন খেতে বলেন. এখন ঔষধ শেষ হওয়ার পর এ সমস্যা যায়নি. বরং ঔষধ খাওয়ার কিছুদিন পর থেকে আমার শ্বাস নিতে কষ্ট হচ্ছে. এর আগে কখনো শ্বাসকষ্ট হয়নি. আমার জ্বর নেই, এখন এই শ্বাস নিতে কষ্ট এবং অতিরিক্ত পালস এর কারনে স্বাভাবিক থাকতে পারছি না আর এমন সংকটময় পরিস্থিতি তে ডাক্তার এর কাছেও যেতে পারছি না. উল্লেখ্য: আমার ঠান্ডার সমস্যা থাকায় অনেক দিন আগে থেকেই Fexofenadine Hydrochloride 120 mg খাই.\"\n"
        "Answer: {\"Age\":[\"22 বছর\"], "
        "\"Symptom\":[\"পালস অনেক বেশি\",\"দুর্বলতা\",\"শ্বাস নিতে কষ্ট\",\"শ্বাসকষ্ট\",\"জ্বর\",\"অতিরিক্ত পালস\",\"ঠান্ডার সমস্যা\"], "
        "\"Medicine\":[\"PROPRANOLOL HCL 10 mg\",\"ALPRAZOLAM 0\",\"Fexofenadine Hydrochloride 120 mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"ECG\",\"ECHO\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বাবা, মা ভাই, বোন কারো এ্যাজমা নেই। আমার এলাজি আছে, বুক ভারি হয়, শ্বাস নিতে সমস্যা হয়। ইকো, ইসিজি, চেস্ট এক্সরে রিপোর্ট নরমাল। আমি এই সমস্যার জন্য কোন ডাক্তার দেখাবো? ডাক্তারভাই এ কি এই বিষয়ে অভিজ্ঞ ডাক্তার আছেন?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"বুক ভারি হয়\",\"শ্বাস নিতে সমস্যা\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"এ্যাজমা\",\"এলাজি\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[\"ইকো\",\"ইসিজি\",\"চেস্ট এক্সরে\"]}\n\n"

    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"she had pain in her left back shoulder then had rolac 10 days then visited a government hospital a doctor prescribed these: Flexi 100mg, Flexllax 10mg, Cosec 20mg, Neurolin 25. Still no change. What specialist doctor should she visit?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"pain in her left back\"], "
        "\"Medicine\":[\"rolac\",\"Flexi\",\"Flexllax\",\"Cosec\",\"Neurolin\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আসসালামু আলাইকুম স্যার, স্যার আমার আম্মুর অস্থিরতা ও শাসকষ্ট, এতে আবার পায়েও পানি আসে। এতে কি হৃদ রোগ হইছে? আর কি করবো বুঝতে পারছি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"অস্থিরতা\",\"শাসকষ্ট\",\"পায়েও পানি আসে\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"হৃদ রোগ\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"এলার্জির সমস্যার কারণে রোদে গেলে গা চিটমিট করে, মাথার ভিতরে কিলবিল করে। সকালে ঘুম থেকে উঠলে অনবরত হাঁচি হয়। কখনো নিয়মিত কোনো এলার্জির ওষুধ খাইনি। এক্ষেত্রে আমি কি করতে পারি? এনার্জির কারণে অনেক দৈনন্দিন কাজ করতে পারি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"রোদে গেলে গা চিটমিট করে\",\"মাথার ভিতরে কিলবিল করে\",\"হাঁচি\"], "
        "\"Medicine\":[\"এলার্জির ওষুধ\"], "
        "\"Health_Condition\":[\"এলার্জির\",\"এনার্জির\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত সোমবার হৈতে আমার হাড়ের জয়েন্টে ব্যথা, গা ম্যাচ ম্যাচ করে। ভিতরে ভিতরে জ্বর লাগে কিন্তু মাপলে ৯৯। নাপা এক্সটা খাইতেছি। উল্লেখ্য ৩/৪ আগে ডিক্স পোলাপ্স ছিল। শূকনো কাশি নাই, তবে টুটি গিলতে হালকা ব্যাথা লাগে। করোনা ভয়ে ivermactin 12mg 2pc ও ডক সিন100 - ৫টি খাইছি। জ্বর সব সময় থাকে না, শরির খুব দুর্লভ।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"হাড়ের জয়েন্টে ব্যথা\",\"গা ম্যাচ ম্যাচ করে\",\"ভিতরে ভিতরে জ্বর লাগে\",\"টুটি গিলতে হালকা ব্যাথা লাগে\",\"জ্বর\",\"শরির খুব দুর্লভ\"], "
        "\"Medicine\":[\"নাপা এক্সটা\",\"ivermactin 12mg\",\"ডক সিন100\"], "
        "\"Health_Condition\":[\"ডিক্স পোলাপ্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত সপ্তাহ থেকে জ্বর আছে, রাতে কাশি বাড়ে, মাঝে মাঝে বমি হয়.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"জ্বর\",\"কাশি\",\"বমি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"বুকে ব্যথা হচ্ছে এবং শ্বাসকষ্ট আছে; খুব অস্বস্তি লাগছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\",\"অস্বস্তি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"রোগী অ্যাজমা ও গ্যাস্ট্রাইটিসের রোগী; গতকাল ধরা পড়েছে মাইগ্রেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"অ্যাজমা\",\"গ্যাস্ট্রাইটিস\",\"মাইগ্রেন\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"থাইরয়েড সমস্যা আছে; history of ডায়াবেটিস উল্লেখ আছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\",\"ডায়াবেটিস\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত চার মাস ধরে মাথা ব্যথা; কাশি না.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"মাথা ব্যথা\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Your serum Triglyceride is slightly raised; HbA1c 6.5%.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত তিন দিন ধরে কাশি ও জ্বর আছে। একজন মেডিসিন বিশেষজ্ঞ আমাকে সেফিক্সিম দিয়েছেন। এক্স-রে করা হয়নি.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"কাশি\",\"জ্বর\"], \"Medicine\":[\"সেফিক্সিম\"], "
        "\"Health_Condition\":[], \"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"রোগীর বয়স ৫৫ বছর। তিনি ডায়াবেটিস ও উচ্চ রক্তচাপের রোগী এবং মেটফরমিন ও লোসারটান খাচ্ছেন.\"\n"
        "Answer: {\"Age\":[\"৫৫ বছর\"], \"Symptom\":[], \"Medicine\":[\"মেটফরমিন\",\"লোসারটান\"], "
        "\"Health_Condition\":[\"ডায়াবেটিস\",\"উচ্চ রক্তচাপ\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"ইএনটি বিশেষজ্ঞ টিম্পানোমেট্রি করতে বলেছেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[\"ইএনটি বিশেষজ্ঞ\"], \"Medical_Procedure\":[\"টিম্পানোমেট্রি\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বাচ্চার ৫০ দিন। দুই তিন দিন যাবত শুকনা ঠান্ডা মনে হচ্ছে। নাক বন্ধ হয়ে থাকে, শ্বাস নিতে কষ্ট হয়। আমি নজোমিষ্ট বিপি০. ৯ % ব্যবহার করছি কিন্তু নাক খোলছে না। এখন আর কি কোন ঔষধ খাওয়াতে হবে আর কি ঔষধ খাওয়াতে পারি??\"\n"
        "Answer: {\"Age\":[\"৫০ দিন\"], \"Symptom\":[\"শুকনা ঠান্ডা\",\"নাক বন্ধ\",\"শ্বাস নিতে কষ্ট\",\"নাক খোলছে না\"], "
        "\"Medicine\":[\"নজোমিষ্ট বিপি০\"], \"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার ছেলের বয়স ১০ বছর। ওর গায়ে পক্স উঠেছিল। এখন ভাল হয়ে গেছে। কিন্তু শরীরে সাদা সাদা দাগ আবার কিছু কাল কাল দাগ দেখা যাচ্ছে। এগুলো কি এমনিতেই চলে যাবে নাকি কোন ওষুধ ব্যবহার করতে হবে?\"\n"
        "Answer: {\"Age\":[\"১০ বছর\"], "
        "\"Symptom\":[\"শরীরে সাদা সাদা দাগ\",\"কাল কাল দাগ\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পক্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),  
]


Translation_based_FEWSHOTS_01: List[str] = [
    (
        "English translation of the text: I have gallstones in my gallbladder. This has been happening for the last 2.5 years. The stone size is 4 cm. I did not have any severe pain for a long time, but within the last 8 days, I had severe stomach pain on 3 days, and I feel pain when I press under the right rib on the right side of my abdomen. If there are gallstones, does abdominal pain occur along with pain when pressing under the right rib on the right side? And why do I feel pain when I press there?? I want advice from an experienced doctor."
        "Text: \"আমার পিত্ত থলিতে পাথর আছে। গত ২.৫ বছর যাবৎ এটা হয়েছে। যার আকার ৪ সেমি। এতদিন তীব্র কোন ব্যাথা ছিল না কিন্তু গত ৮ দিনের মধ্যে ৩ দিন পেটে তীব্র ব্যাথা হয়েছিল এবং পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়। পিত্ত থলিতে পাথর হলে কী পেট ব্যাথার সাথে সাথে পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়? এবং কেন এই চাপ দিলে ব্যাথা অনুভূত হয়?? অভিজ্ঞ ডাক্তারের পরামর্শ চাচ্ছি।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"ব্যাথা\",\"পেটে তীব্র ব্যাথা\",\"চাপ দিলে ব্যাথা\",\"পেট ব্যাথার\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পিত্ত থলিতে পাথর\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: Assalamu alaikum, my age is 22 years. For the last one and a half months my pulse has been very high (most of the time above 100), especially after eating. Because I also felt weakness, a month ago I saw a medicine specialist. He asked me to do ECG and ECHO, but since the report did not show anything abnormal, he told me to take PROPRANOLOL HCL 10 mg for 30 days and ALPRAZOLAM 0.25 mg for 10 days. After finishing the medicines, the problem did not go away. Rather, after taking the medicines for a few days, I started having difficulty breathing. I never had shortness of breath before. I have no fever, but now I cannot stay normal due to this breathing difficulty and excessive pulse, and in this critical situation I also cannot go to a doctor. Note: Because I have a cold/allergy issue, I have been taking Fexofenadine Hydrochloride 120 mg for a long time."
        "Text: \"আস্সালামুয়ালাইকুম, আমার বয়স 22 বছর. গত দেড় মাস যাবত আমার পালস অনেক বেশি থাকে ( বেশি সময় 100 এর উপরে ) বিশেষ করে খাবার খাওয়ার পর. সাথে দুর্বলতা থাকায় এক মাস আগে আমি একজন MEDিসিন বিশেষজ্ঞ কে দেখাই. তিনি ECG এবং ECHO করাতে বলেন কিন্তু রিপোর্টে তেমন অস্বাভাবিক কিছু না থাকায় তিনি আমাকে PROPRANOLOL HCL 10 mg ৩০ দিন এবং ALPRAZOLAM 0. 25 mg ১০ দিন খেতে বলেন. এখন ঔষধ শেষ হওয়ার পর এ সমস্যা যায়নি. বরং ঔষধ খাওয়ার কিছুদিন পর থেকে আমার শ্বাস নিতে কষ্ট হচ্ছে. এর আগে কখনো শ্বাসকষ্ট হয়নি. আমার জ্বর নেই, এখন এই শ্বাস নিতে কষ্ট এবং অতিরিক্ত পালস এর কারনে স্বাভাবিক থাকতে পারছি না আর এমন সংকটময় পরিস্থিতি তে ডাক্তার এর কাছেও যেতে পারছি না. উল্লেখ্য: আমার ঠান্ডার সমস্যা থাকায় অনেক দিন আগে থেকেই Fexofenadine Hydrochloride 120 mg খাই.\"\n"
        "Answer: {\"Age\":[\"22 বছর\"], "
        "\"Symptom\":[\"পালস অনেক বেশি\",\"দুর্বলতা\",\"শ্বাস নিতে কষ্ট\",\"শ্বাসকষ্ট\",\"জ্বর\",\"অতিরিক্ত পালস\",\"ঠান্ডার সমস্যা\"], "
        "\"Medicine\":[\"PROPRANOLOL HCL 10 mg\",\"ALPRAZOLAM 0\",\"Fexofenadine Hydrochloride 120 mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"ECG\",\"ECHO\"]}\n\n"
    ),
    (
        "English translation of the text: None of my father, mother, brother, or sister has asthma. I have allergy, my chest feels heavy, and I have trouble breathing. My echo, ECG, and chest X-ray reports are normal. For this problem, which doctor should I see? Is there any experienced doctor for this issue?"
        "Text: \"আমার বাবা, মা ভাই, বোন কারো এ্যাজমা নেই। আমার এলাজি আছে, বুক ভারি হয়, শ্বাস নিতে সমস্যা হয়। ইকো, ইসিজি, চেস্ট এক্সরে রিপোর্ট নরমাল। আমি এই সমস্যার জন্য কোন ডাক্তার দেখাবো? ডাক্তারভাই এ কি এই বিষয়ে অভিজ্ঞ ডাক্তার আছেন?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"বুক ভারি হয়\",\"শ্বাস নিতে সমস্যা\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"এ্যাজমা\",\"এলাজি\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[\"ইকো\",\"ইসিজি\",\"চেস্ট এক্সরে\"]}\n\n"
    ),
    (
        "English translation of the text: She had pain in her left back shoulder, then took Rolac for 10 days, then visited a government hospital. A doctor prescribed these medicines: Flexi 100mg, Flexllax 10mg, Cosec 20mg, Neurolin 25. Still no change. What specialist doctor should she visit?"
        "Text: \"she had pain in her left back shoulder then had rolac 10 days then visited a government hospital a doctor prescribed these: Flexi 100mg, Flexllax 10mg, Cosec 20mg, Neurolin 25. Still no change. What specialist doctor should she visit?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"pain in her left back\"], "
        "\"Medicine\":[\"rolac\",\"Flexi\",\"Flexllax\",\"Cosec\",\"Neurolin\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: Assalamu alaikum sir, my mother has restlessness and shortness of breath, and her legs also swell with water. Has she developed heart disease? And what should I do, I cannot understand."
        "Text: \"আসসালামু আলাইকুম স্যার, স্যার আমার আম্মুর অস্থিরতা ও শাসকষ্ট, এতে আবার পায়েও পানি আসে। এতে কি হৃদ রোগ হইছে? আর কি করবো বুঝতে পারছি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"অস্থিরতা\",\"শাসকষ্ট\",\"পায়েও পানি আসে\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"হৃদ রোগ\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: Because of allergy problems, when I go out in the sun my body tingles, and I feel a crawling sensation inside my head. In the morning after waking up I sneeze continuously. I have never taken any allergy medicine regularly. In this case, what can I do? Because of this allergy, I cannot do many daily tasks."
        "Text: \"এলার্জির সমস্যার কারণে রোদে গেলে গা চিটমিট করে, মাথার ভিতরে কিলবিল করে। সকালে ঘুম থেকে উঠলে অনবরত হাঁচি হয়। কখনো নিয়মিত কোনো এলার্জির ওষুধ খাইনি। এক্ষেত্রে আমি কি করতে পারি? এনার্জির কারণে অনেক দৈনন্দিন কাজ করতে পারি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"রোদে গেলে গা চিটমিট করে\",\"মাথার ভিতরে কিলবিল করে\",\"হাঁচি\"], "
        "\"Medicine\":[\"এলার্জির ওষুধ\"], "
        "\"Health_Condition\":[\"এলার্জির\",\"এনার্জির\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: Since last Monday I have pain in the joints of my bones and body aches. I feel feverish inside, but when I measure it, it is 99. I am taking Napa Extra. Note: 3–4 years ago I had disc prolapse. I do not have a dry cough, but I have mild pain when swallowing. Out of fear of coronavirus I took ivermactin 12mg 2 pieces and Doc Sin100 - 5 tablets. The fever is not always there, and my body is very weak."
        "Text: \"গত সোমবার হৈতে আমার হাড়ের জয়েন্টে ব্যথা, গা ম্যাচ ম্যাচ করে। ভিতরে ভিতরে জ্বর লাগে কিন্তু মাপলে ৯৯। নাপা এক্সটা খাইতেছি। উল্লেখ্য ৩/৪ আগে ডিক্স পোলাপ্স ছিল। শূকনো কাশি নাই, তবে টুটি গিলতে হালকা ব্যাথা লাগে। করোনা ভয়ে ivermactin 12mg 2pc ও ডক সিন100 - ৫টি খাইছি। জ্বর সব সময় থাকে না, শরির খুব দুর্লভ।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"হাড়ের জয়েন্টে ব্যথা\",\"গা ম্যাচ ম্যাচ করে\",\"ভিতরে ভিতরে জ্বর লাগে\",\"টুটি গিলতে হালকা ব্যাথা লাগে\",\"জ্বর\",\"শরির খুব দুর্লভ\"], "
        "\"Medicine\":[\"নাপা এক্সটা\",\"ivermactin 12mg\",\"ডক সিন100\"], "
        "\"Health_Condition\":[\"ডিক্স পোলাপ্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: I have had fever since last week, the cough gets worse at night, and sometimes I vomit."
        "Text: \"গত সপ্তাহ থেকে জ্বর আছে, রাতে কাশি বাড়ে, মাঝে মাঝে বমি হয়.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"জ্বর\",\"কাশি\",\"বমি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: I have chest pain and shortness of breath; I feel very uncomfortable."
        "Text: \"বুকে ব্যথা হচ্ছে এবং শ্বাসকষ্ট আছে; খুব অস্বস্তি লাগছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\",\"অস্বস্তি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: The patient has asthma and gastritis; yesterday migraine was diagnosed."
        "Text: \"রোগী অ্যাজমা ও গ্যাস্ট্রাইটিসের রোগী; গতকাল ধরা পড়েছে মাইগ্রেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"অ্যাজমা\",\"গ্যাস্ট্রাইটিস\",\"মাইগ্রেন\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: There is a thyroid problem; a history of diabetes is mentioned."
        "Text: \"থাইরয়েড সমস্যা আছে; history of ডায়াবেটিস উল্লেখ আছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\",\"ডায়াবেটিস\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: For the last four months I have had a headache; no cough."
        "Text: \"গত চার মাস ধরে মাথা ব্যথা; কাশি না.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"মাথা ব্যথা\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: Your serum triglyceride is slightly raised; HbA1c is 6.5%."
        "Text: \"Your serum Triglyceride is slightly raised; HbA1c 6.5%.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: I have had a cough and fever for the past three days. A medicine specialist prescribed Cefixime for me. An X-ray has not been done."
        "Text: \"গত তিন দিন ধরে কাশি ও জ্বর আছে। একজন মেডিসিন বিশেষজ্ঞ আমাকে সেফিক্সিম দিয়েছেন। এক্স-রে করা হয়নি.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"কাশি\",\"জ্বর\"], \"Medicine\":[\"সেফিক্সিম\"], "
        "\"Health_Condition\":[], \"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: The patient is 55 years old. He has diabetes and high blood pressure and is taking metformin and losartan."
        "Text: \"রোগীর বয়স ৫৫ বছর। তিনি ডায়াবেটিস ও উচ্চ রক্তচাপের রোগী এবং মেটফরমিন ও লোসারটান খাচ্ছেন.\"\n"
        "Answer: {\"Age\":[\"৫৫ বছর\"], \"Symptom\":[], \"Medicine\":[\"মেটফরমিন\",\"লোসারটান\"], "
        "\"Health_Condition\":[\"ডায়াবেটিস\",\"উচ্চ রক্তচাপ\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: An ENT specialist has asked to do tympanometry."
        "Text: \"ইএনটি বিশেষজ্ঞ টিম্পানোমেট্রি করতে বলেছেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[\"ইএনটি বিশেষজ্ঞ\"], \"Medical_Procedure\":[\"টিম্পানোমেট্রি\"]}\n\n"
    ),
    (
        "English translation of the text: My baby is 50 days old. For the last two or three days it seems like a dry cold. The nose stays blocked and there is difficulty breathing. I am using Nazomist BP 0.9% but the nose is not opening. Now do I need to give any medicine to eat, and what medicine can I give?"
        "Text: \"আমার বাচ্চার ৫০ দিন। দুই তিন দিন যাবত শুকনা ঠান্ডা মনে হচ্ছে। নাক বন্ধ হয়ে থাকে, শ্বাস নিতে কষ্ট হয়। আমি নজোমিষ্ট বিপি০. ৯ % ব্যবহার করছি কিন্তু নাক খোলছে না। এখন আর কি কোন ঔষধ খাওয়াতে হবে আর কি ঔষধ খাওয়াতে পারি??\"\n"
        "Answer: {\"Age\":[\"৫০ দিন\"], \"Symptom\":[\"শুকনা ঠান্ডা\",\"নাক বন্ধ\",\"শ্বাস নিতে কষ্ট\",\"নাক খোলছে না\"], "
        "\"Medicine\":[\"নজোমিষ্ট বিপি০\"], \"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "English translation of the text: My son is 10 years old. He had pox on his body. Now he has recovered. But white spots on the body and some dark spots are now appearing again. Will these go away on their own or do I need to use any medicine?"
        "Text: \"আমার ছেলের বয়স ১০ বছর। ওর গায়ে পক্স উঠেছিল। এখন ভাল হয়ে গেছে। কিন্তু শরীরে সাদা সাদা দাগ আবার কিছু কাল কাল দাগ দেখা যাচ্ছে। এগুলো কি এমনিতেই চলে যাবে নাকি কোন ওষুধ ব্যবহার করতে হবে?\"\n"
        "Answer: {\"Age\":[\"১০ বছর\"], "
        "\"Symptom\":[\"শরীরে সাদা সাদা দাগ\",\"কাল কাল দাগ\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পক্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
]


QA_FEWSHOTS_02: List[str] = [
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আসসালামু আলাইকুম ডাক্তার, আমার বয়স ৩৪ বছর। গত দুই সপ্তাহ ধরে বুক ধড়ফড় করে, মাথা ঘোরে এবং মাঝে মাঝে শ্বাস নিতে কষ্ট হয়। স্থানীয় কার্ডিওলজিস্ট আমাকে ECG, ECHO এবং ট্রোপোনিন টেস্ট করতে বলেছেন। তিনি বললেন উচ্চ রক্তচাপ ও অ্যানজাইটি থাকতে পারে। আমি Atenolol 25 mg আর Clonazepam 0.5 mg খাচ্ছি কিন্তু রাতে ঘুম কম হয়।\"\n"
        "Answer: {\"Age\":[\"৩৪ বছর\"], "
        "\"Symptom\":[\"বুক ধড়ফড়\",\"মাথা ঘোরে\",\"শ্বাস নিতে কষ্ট\",\"ঘুম কম\"], "
        "\"Medicine\":[\"Atenolol 25 mg\",\"Clonazepam 0.5 mg\"], "
        "\"Health_Condition\":[\"উচ্চ রক্তচাপ\",\"অ্যানজাইটি\"], "
        "\"Specialist\":[\"কার্ডিওলজিস্ট\"], "
        "\"Medical_Procedure\":[\"ECG\",\"ECHO\",\"ট্রোপোনিন টেস্ট\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"I am 28 years old and for the last 3 months I have frequent heartburn, nausea, and a burning pain after meals. A gastroenterologist suggested an endoscopy and H. pylori test. He thinks it could be gastritis or GERD. I was prescribed Omeprazole 20mg and Sucralfate syrup, but the bloating and sour belching still returns at night.\"\n"
        "Answer: {\"Age\":[\"28 years old\"], "
        "\"Symptom\":[\"heartburn\",\"nausea\",\"burning pain after meals\",\"bloating\",\"sour belching\"], "
        "\"Medicine\":[\"Omeprazole 20mg\",\"Sucralfate syrup\"], "
        "\"Health_Condition\":[\"gastritis\",\"GERD\"], "
        "\"Specialist\":[\"gastroenterologist\"], "
        "\"Medical_Procedure\":[\"endoscopy\",\"H. pylori test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Paciente de 52 años con dolor en las rodillas, rigidez matutina y hinchazón en las manos desde hace 6 semanas. El reumatólogo pidió análisis de factor reumatoide, PCR y una radiografía de manos. Sospecha artritis reumatoide. Me indicó Naproxeno 500 mg y Prednisona 5 mg, pero sigo con fatiga y dolor al caminar.\"\n"
        "Answer: {\"Age\":[\"52 años\"], "
        "\"Symptom\":[\"dolor en las rodillas\",\"rigidez matutina\",\"hinchazón en las manos\",\"fatiga\",\"dolor al caminar\"], "
        "\"Medicine\":[\"Naproxeno 500 mg\",\"Prednisona 5 mg\"], "
        "\"Health_Condition\":[\"artritis reumatoide\"], "
        "\"Specialist\":[\"reumatólogo\"], "
        "\"Medical_Procedure\":[\"análisis de factor reumatoide\",\"PCR\",\"radiografía de manos\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার ছেলের বয়স ৭ বছর। গত এক মাস ধরে বারবার হাঁচি, নাক দিয়ে পানি পড়া, চোখ চুলকানি এবং রাতে কাশি বাড়ে। একজন অ্যালার্জি বিশেষজ্ঞ স্কিন প্রিক টেস্ট ও Spirometry করতে বলেছেন এবং অ্যালার্জিক রাইনাইটিস ও অ্যাজমা সন্দেহ করছেন। তিনি Montelukast 4 mg ও Cetirizine syrup দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"৭ বছর\"], "
        "\"Symptom\":[\"হাঁচি\",\"নাক দিয়ে পানি পড়া\",\"চোখ চুলকানি\",\"কাশি\"], "
        "\"Medicine\":[\"Montelukast 4 mg\",\"Cetirizine syrup\"], "
        "\"Health_Condition\":[\"অ্যালার্জিক রাইনাইটিস\",\"অ্যাজমা\"], "
        "\"Specialist\":[\"অ্যালার্জি বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"স্কিন প্রিক টেস্ট\",\"Spirometry\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"She is 19 years old and has severe lower abdominal pain, burning urination, and fever since yesterday. The gynecologist asked for urine culture and pelvic ultrasound and said it might be a UTI or ovarian cyst. She started Nitrofurantoin 100mg and Paracetamol, but the pain is still sharp when she moves.\"\n"
        "Answer: {\"Age\":[\"19 years old\"], "
        "\"Symptom\":[\"severe lower abdominal pain\",\"burning urination\",\"fever\",\"sharp pain\"], "
        "\"Medicine\":[\"Nitrofurantoin 100mg\",\"Paracetamol\"], "
        "\"Health_Condition\":[\"UTI\",\"ovarian cyst\"], "
        "\"Specialist\":[\"gynecologist\"], "
        "\"Medical_Procedure\":[\"urine culture\",\"pelvic ultrasound\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ৪৬ বছর এবং অনেকদিন ধরে কোমরের নিচে ব্যথা ডান পায়ে ছড়ায়, পা ঝিনঝিন করে এবং হাটতে গেলে ব্যথা বাড়ে। একজন অর্থোপেডিক বিশেষজ্ঞ MRI (লম্বার স্পাইন) ও X-ray করতে বলেছেন এবং সায়াটিকা ও ডিস্ক প্রোলাপ্স বলেছেন। তিনি Gabapentin 300 mg আর Diclofenac gel ব্যবহার করতে বলেছেন।\"\n"
        "Answer: {\"Age\":[\"৪৬ বছর\"], "
        "\"Symptom\":[\"কোমরের নিচে ব্যথা\",\"ডান পায়ে ছড়ায়\",\"পা ঝিনঝিন\",\"হাটতে গেলে ব্যথা বাড়ে\"], "
        "\"Medicine\":[\"Gabapentin 300 mg\",\"Diclofenac gel\"], "
        "\"Health_Condition\":[\"সায়াটিকা\",\"ডিস্ক প্রোলাপ্স\"], "
        "\"Specialist\":[\"অর্থোপেডিক বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"MRI (লম্বার স্পাইন)\",\"X-ray\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 33 años y desde hace una semana tengo tos seca, dolor de garganta, congestión nasal y cansancio. El médico general pidió una prueba de COVID, hemograma y una radiografía de tórax. Sospecha bronquitis. Me recetó Azitromicina 500 mg y Jarabe de ambroxol, pero la tos empeora por la noche.\"\n"
        "Answer: {\"Age\":[\"33 años\"], "
        "\"Symptom\":[\"tos seca\",\"dolor de garganta\",\"congestión nasal\",\"cansancio\",\"la tos empeora por la noche\"], "
        "\"Medicine\":[\"Azitromicina 500 mg\",\"Jarabe de ambroxol\"], "
        "\"Health_Condition\":[\"bronquitis\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[\"prueba de COVID\",\"hemograma\",\"radiografía de tórax\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My father is 61 years old and has numbness in both feet, blurry vision, and excessive thirst for months. The endocrinologist ordered fasting blood sugar, HbA1c, and a urine microalbumin test. He said it looks like diabetes with possible neuropathy. He started Metformin 500mg and Insulin glargine at night.\"\n"
        "Answer: {\"Age\":[\"61 years old\"], "
        "\"Symptom\":[\"numbness in both feet\",\"blurry vision\",\"excessive thirst\"], "
        "\"Medicine\":[\"Metformin 500mg\",\"Insulin glargine\"], "
        "\"Health_Condition\":[\"diabetes\",\"neuropathy\"], "
        "\"Specialist\":[\"endocrinologist\"], "
        "\"Medical_Procedure\":[\"fasting blood sugar\",\"HbA1c\",\"urine microalbumin test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ২৭ বছর এবং কয়েকদিন ধরে তীব্র মাথাব্যথা, আলো সহ্য হয় না, বমি বমি ভাব এবং মাঝে মাঝে বমি হচ্ছে। একজন নিউরোলজিস্ট CT scan ও চোখের ফান্ডাস পরীক্ষা করতে বলেছেন এবং মাইগ্রেন বলেছেন। তিনি Sumatriptan 50 mg ও Domperidone দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"২৭ বছর\"], "
        "\"Symptom\":[\"তীব্র মাথাব্যথা\",\"আলো সহ্য হয় না\",\"বমি বমি ভাব\",\"বমি\"], "
        "\"Medicine\":[\"Sumatriptan 50 mg\",\"Domperidone\"], "
        "\"Health_Condition\":[\"মাইগ্রেন\"], "
        "\"Specialist\":[\"নিউরোলজিস্ট\"], "
        "\"Medical_Procedure\":[\"CT scan\",\"চোখের ফান্ডাস পরীক্ষা\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"She is 40 years old with recurrent skin rashes, itching all over the body, and swelling of lips after eating shrimp. The dermatologist advised an IgE test and a patch test, and said it is likely food allergy with urticaria. She was given Fexofenadine 120mg and Hydrocortisone cream.\"\n"
        "Answer: {\"Age\":[\"40 years old\"], "
        "\"Symptom\":[\"recurrent skin rashes\",\"itching all over the body\",\"swelling of lips\"], "
        "\"Medicine\":[\"Fexofenadine 120mg\",\"Hydrocortisone cream\"], "
        "\"Health_Condition\":[\"food allergy\",\"urticaria\"], "
        "\"Specialist\":[\"dermatologist\"], "
        "\"Medical_Procedure\":[\"IgE test\",\"patch test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার মায়ের বয়স ৫৮ বছর। তিনি অনেকদিন ধরে হাঁটুতে ব্যথা, উঠতে বসতে কষ্ট এবং পা ফুলে যায়। একজন অর্থোপেডিক বিশেষজ্ঞ X-ray (হাঁটু) ও ভিটামিন ডি টেস্ট করতে বলেছেন এবং অস্টিওআর্থ্রাইটিস বলেছেন। তিনি Calcium + Vitamin D ট্যাবলেট এবং Aceclofenac 100 mg খেতে বলেছেন।\"\n"
        "Answer: {\"Age\":[\"৫৮ বছর\"], "
        "\"Symptom\":[\"হাঁটুতে ব্যথা\",\"উঠতে বসতে কষ্ট\",\"পা ফুলে যায়\"], "
        "\"Medicine\":[\"Calcium + Vitamin D\",\"Aceclofenac 100 mg\"], "
        "\"Health_Condition\":[\"অস্টিওআর্থ্রাইটিস\"], "
        "\"Specialist\":[\"অর্থোপেডিক বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"X-ray (হাঁটু)\",\"ভিটামিন ডি টেস্ট\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 24 años y siento dolor al orinar, urgencia urinaria y dolor en la parte baja del abdomen. La ginecóloga solicitó examen de orina y cultivo de orina, y sospecha cistitis. Me dio Fosfomicina 3 g y Ibuprofeno, pero todavía tengo ardor por la tarde.\"\n"
        "Answer: {\"Age\":[\"24 años\"], "
        "\"Symptom\":[\"dolor al orinar\",\"urgencia urinaria\",\"dolor en la parte baja del abdomen\",\"ardor\"], "
        "\"Medicine\":[\"Fosfomicina 3 g\",\"Ibuprofeno\"], "
        "\"Health_Condition\":[\"cistitis\"], "
        "\"Specialist\":[\"ginecóloga\"], "
        "\"Medical_Procedure\":[\"examen de orina\",\"cultivo de orina\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My brother is 12 years old, has persistent sore throat, snoring at night, and difficulty swallowing. The ENT specialist recommended a throat swab culture and a neck ultrasound, and mentioned tonsillitis with possible enlarged adenoids. He prescribed Amoxicillin 250mg and a saline gargle.\"\n"
        "Answer: {\"Age\":[\"12 years old\"], "
        "\"Symptom\":[\"persistent sore throat\",\"snoring at night\",\"difficulty swallowing\"], "
        "\"Medicine\":[\"Amoxicillin 250mg\",\"saline gargle\"], "
        "\"Health_Condition\":[\"tonsillitis\",\"enlarged adenoids\"], "
        "\"Specialist\":[\"ENT specialist\"], "
        "\"Medical_Procedure\":[\"throat swab culture\",\"neck ultrasound\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ৩১ বছর এবং গত তিন মাস ধরে অনিয়মিত পিরিয়ড, তলপেটে ব্যথা এবং অতিরিক্ত চুল পড়ছে। একজন গাইনী বিশেষজ্ঞ Hormone profile, TSH টেস্ট ও Pelvic ultrasound করতে বলেছেন এবং PCOS ও থাইরয়েড সমস্যা সন্দেহ করছেন। তিনি Metformin 500 mg ও Levothyroxine 50 mcg দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"৩১ বছর\"], "
        "\"Symptom\":[\"অনিয়মিত পিরিয়ড\",\"তলপেটে ব্যথা\",\"অতিরিক্ত চুল পড়ছে\"], "
        "\"Medicine\":[\"Metformin 500 mg\",\"Levothyroxine 50 mcg\"], "
        "\"Health_Condition\":[\"PCOS\",\"থাইরয়েড সমস্যা\"], "
        "\"Specialist\":[\"গাইনী বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"Hormone profile\",\"TSH টেস্ট\",\"Pelvic ultrasound\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 67 años con falta de aire al caminar, hinchazón en los tobillos y tos nocturna. El cardiólogo solicitó BNP, ECG y ecocardiograma. Dijo que puede ser insuficiencia cardíaca. Me indicó Furosemida 40 mg y Enalapril 10 mg, pero sigo cansado y con opresión en el pecho.\"\n"
        "Answer: {\"Age\":[\"67 años\"], "
        "\"Symptom\":[\"falta de aire al caminar\",\"hinchazón en los tobillos\",\"tos nocturna\",\"cansado\",\"opresión en el pecho\"], "
        "\"Medicine\":[\"Furosemida 40 mg\",\"Enalapril 10 mg\"], "
        "\"Health_Condition\":[\"insuficiencia cardíaca\"], "
        "\"Specialist\":[\"cardiólogo\"], "
        "\"Medical_Procedure\":[\"BNP\",\"ECG\",\"ecocardiograma\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My mother is 45 years old and has severe tooth pain, gum swelling, and fever since last night. The dentist advised a dental X-ray and said it could be an abscess with infection. She started Amoxiclav 625mg and Diclofenac, but she still cannot chew on that side.\"\n"
        "Answer: {\"Age\":[\"45 years old\"], "
        "\"Symptom\":[\"severe tooth pain\",\"gum swelling\",\"fever\",\"cannot chew on that side\"], "
        "\"Medicine\":[\"Amoxiclav 625mg\",\"Diclofenac\"], "
        "\"Health_Condition\":[\"abscess\",\"infection\"], "
        "\"Specialist\":[\"dentist\"], "
        "\"Medical_Procedure\":[\"dental X-ray\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ৬০ বছর এবং দুই সপ্তাহ ধরে প্রস্রাবে জ্বালা, ঘন ঘন প্রস্রাব, তলপেটে চাপ অনুভব এবং মাঝে মাঝে জ্বর হচ্ছে। একজন ইউরোলজি বিশেষজ্ঞ Urine R/E, Urine culture ও USG KUB করতে বলেছেন এবং ইউটিআই বলেছেন। তিনি Cefixime 200 mg ও Phenazopyridine দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"৬০ বছর\"], "
        "\"Symptom\":[\"প্রস্রাবে জ্বালা\",\"ঘন ঘন প্রস্রাব\",\"তলপেটে চাপ অনুভব\",\"জ্বর\"], "
        "\"Medicine\":[\"Cefixime 200 mg\",\"Phenazopyridine\"], "
        "\"Health_Condition\":[\"ইউটিআই\"], "
        "\"Specialist\":[\"ইউরোলজি বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"Urine R/E\",\"Urine culture\",\"USG KUB\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 38 años y desde hace meses tengo dolor de cabeza frecuente, visión borrosa y presión alta en controles. El neurólogo pidió una resonancia magnética cerebral y un examen de fondo de ojo, y también sugirió medir la presión arterial 24 horas. Me recetó Losartán 50 mg y Amitriptilina 10 mg para el dolor, pero sigo con mareos.\"\n"
        "Answer: {\"Age\":[\"38 años\"], "
        "\"Symptom\":[\"dolor de cabeza frecuente\",\"visión borrosa\",\"presión alta\",\"mareos\"], "
        "\"Medicine\":[\"Losartán 50 mg\",\"Amitriptilina 10 mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"neurólogo\"], "
        "\"Medical_Procedure\":[\"resonancia magnética cerebral\",\"examen de fondo de ojo\",\"medir la presión arterial 24 horas\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My son is 6 years old and has repeated wheezing, chest tightness, and shortness of breath during play. The pediatric pulmonologist ordered a chest X-ray and spirometry and said it may be asthma. He prescribed Salbutamol inhaler and Budesonide nebulization, but the wheeze still comes back after running.\"\n"
        "Answer: {\"Age\":[\"6 years old\"], "
        "\"Symptom\":[\"repeated wheezing\",\"chest tightness\",\"shortness of breath\",\"wheeze still comes back\"], "
        "\"Medicine\":[\"Salbutamol inhaler\",\"Budesonide nebulization\"], "
        "\"Health_Condition\":[\"asthma\"], "
        "\"Specialist\":[\"pediatric pulmonologist\"], "
        "\"Medical_Procedure\":[\"chest X-ray\",\"spirometry\"]}\n\n"
    ),
]

QA_FEWSHOTS_03: List[str] = [
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ৩৪ বছর। তিন দিন ধরে জ্বর আর কাশি হচ্ছে। ডাক্তার CBC এবং Chest X-ray করতে বলেছেন এবং Azithromycin 500mg খেতে বলেছেন।\"\n"
        "Answer: {\"Age\":[\"৩৪ বছর\"], "
        "\"Symptom\":[\"জ্বর\",\"কাশি\"], "
        "\"Medicine\":[\"Azithromycin 500mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"ডাক্তার\"], "
        "\"Medical_Procedure\":[\"CBC\",\"Chest X-ray\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Patient age 67 years. He has diabetes and hypertension. Taking Metformin and Amlodipine. Cardiologist advised ECG and Echocardiogram.\"\n"
        "Answer: {\"Age\":[\"67 years\"], "
        "\"Symptom\":[], "
        "\"Medicine\":[\"Metformin\",\"Amlodipine\"], "
        "\"Health_Condition\":[\"diabetes\",\"hypertension\"], "
        "\"Specialist\":[\"Cardiologist\"], "
        "\"Medical_Procedure\":[\"ECG\",\"Echocardiogram\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 29 años. Desde ayer tengo dolor de garganta y fiebre. El médico me dio Paracetamol 500 mg y pidió una prueba PCR.\"\n"
        "Answer: {\"Age\":[\"29 años\"], "
        "\"Symptom\":[\"dolor de garganta\",\"fiebre\"], "
        "\"Medicine\":[\"Paracetamol 500 mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"médico\"], "
        "\"Medical_Procedure\":[\"prueba PCR\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Aitaren adina ৫৮ বছর। বুকে ব্যথা আর শ্বাসকষ্ট হচ্ছে। Cardiologist angiography করতে বলেছে এবং aspirin খেতে বলেছে।\"\n"
        "Answer: {\"Age\":[\"৫৮ বছর\"], "
        "\"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\"], "
        "\"Medicine\":[\"aspirin\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"Cardiologist\"], "
        "\"Medical_Procedure\":[\"angiography\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার মেয়ের বয়স ১২ বছর। এলার্জি আছে, নাক দিয়ে পানি পড়ে ও হাঁচি হয়। ENT বিশেষজ্ঞ Skin prick test করতে বলেছেন এবং Cetirizine খেতে বলেছেন।\"\n"
        "Answer: {\"Age\":[\"১২ বছর\"], "
        "\"Symptom\":[\"নাক দিয়ে পানি পড়ে\",\"হাঁচি\"], "
        "\"Medicine\":[\"Cetirizine\"], "
        "\"Health_Condition\":[\"এলার্জি\"], "
        "\"Specialist\":[\"ENT বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"Skin prick test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My son is 8 years old. He has abdominal pain and vomiting. The pediatrician ordered an ultrasound and prescribed ORS and ondansetron.\"\n"
        "Answer: {\"Age\":[\"8 years old\"], "
        "\"Symptom\":[\"abdominal pain\",\"vomiting\"], "
        "\"Medicine\":[\"ORS\",\"ondansetron\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"pediatrician\"], "
        "\"Medical_Procedure\":[\"ultrasound\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ২৬ বছর। মাথা ঘোরে আর দুর্বল লাগে। Neurologist MRI করতে বলেছে এবং Vitamin B12 ইনজেকশন দিতে বলেছে।\"\n"
        "Answer: {\"Age\":[\"২৬ বছর\"], "
        "\"Symptom\":[\"মাথা ঘোরে\",\"দুর্বল লাগে\"], "
        "\"Medicine\":[\"Vitamin B12 ইনজেকশন\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"Neurologist\"], "
        "\"Medical_Procedure\":[\"MRI\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 45 años y sufro de asma. Hoy tengo dificultad para respirar y tos. El neumólogo me indicó nebulización y salbutamol.\"\n"
        "Answer: {\"Age\":[\"45 años\"], "
        "\"Symptom\":[\"dificultad para respirar\",\"tos\"], "
        "\"Medicine\":[\"salbutamol\"], "
        "\"Health_Condition\":[\"asma\"], "
        "\"Specialist\":[\"neumólogo\"], "
        "\"Medical_Procedure\":[\"nebulización\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"বয়স ৬০ বছর। কিডনিতে পাথর ধরা পড়েছে। Urologist CT scan করতে বলেছে এবং Tamsulosin 0.4mg দিয়েছে।\"\n"
        "Answer: {\"Age\":[\"৬০ বছর\"], "
        "\"Symptom\":[], "
        "\"Medicine\":[\"Tamsulosin 0.4mg\"], "
        "\"Health_Condition\":[\"কিডনিতে পাথর\"], "
        "\"Specialist\":[\"Urologist\"], "
        "\"Medical_Procedure\":[\"CT scan\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My mother is 52 years old. She has high blood pressure and headache. The physician prescribed Losartan and advised blood pressure monitoring.\"\n"
        "Answer: {\"Age\":[\"52 years old\"], "
        "\"Symptom\":[\"headache\"], "
        "\"Medicine\":[\"Losartan\"], "
        "\"Health_Condition\":[\"high blood pressure\"], "
        "\"Specialist\":[\"physician\"], "
        "\"Medical_Procedure\":[\"blood pressure monitoring\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বাচ্চার বয়স ৩ মাস। ডায়রিয়া হচ্ছে আর পানিশূন্যতা মনে হচ্ছে। Pediatrician stool test করতে বলেছেন এবং Zinc syrup দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"৩ মাস\"], "
        "\"Symptom\":[\"ডায়রিয়া\",\"পানিশূন্যতা\"], "
        "\"Medicine\":[\"Zinc syrup\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"Pediatrician\"], "
        "\"Medical_Procedure\":[\"stool test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Tengo 33 años. Me duele el pecho y siento palpitaciones. El cardiólogo solicitó un Holter y recetó propranolol.\"\n"
        "Answer: {\"Age\":[\"33 años\"], "
        "\"Symptom\":[\"Me duele el pecho\",\"palpitaciones\"], "
        "\"Medicine\":[\"propranolol\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"cardiólogo\"], "
        "\"Medical_Procedure\":[\"Holter\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"রোগীর বয়স ৪০ বছর। গ্যাস্ট্রিক সমস্যা আছে এবং বুক জ্বালা করে। Gastroenterologist endoscopy করতে বলেছেন এবং Omeprazole 20mg দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"৪০ বছর\"], "
        "\"Symptom\":[\"বুক জ্বালা করে\"], "
        "\"Medicine\":[\"Omeprazole 20mg\"], "
        "\"Health_Condition\":[\"গ্যাস্ট্রিক সমস্যা\"], "
        "\"Specialist\":[\"Gastroenterologist\"], "
        "\"Medical_Procedure\":[\"endoscopy\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"My age is 24. I have skin rash and itching. The dermatologist advised allergy test and prescribed hydrocortisone cream.\"\n"
        "Answer: {\"Age\":[\"24\"], "
        "\"Symptom\":[\"skin rash\",\"itching\"], "
        "\"Medicine\":[\"hydrocortisone cream\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"dermatologist\"], "
        "\"Medical_Procedure\":[\"allergy test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ১৮ বছর। গলা ব্যথা ও নাক বন্ধ। ENT doctor rapid antigen test করতে বলেছে এবং Amoxicillin খেতে দিয়েছে।\"\n"
        "Answer: {\"Age\":[\"১৮ বছর\"], "
        "\"Symptom\":[\"গলা ব্যথা\",\"নাক বন্ধ\"], "
        "\"Medicine\":[\"Amoxicillin\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"ENT doctor\"], "
        "\"Medical_Procedure\":[\"rapid antigen test\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"Age 71 years. She has arthritis and joint pain. Rheumatologist suggested physiotherapy and prescribed naproxen.\"\n"
        "Answer: {\"Age\":[\"71 years\"], "
        "\"Symptom\":[\"joint pain\"], "
        "\"Medicine\":[\"naproxen\"], "
        "\"Health_Condition\":[\"arthritis\"], "
        "\"Specialist\":[\"Rheumatologist\"], "
        "\"Medical_Procedure\":[\"physiotherapy\"]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"আমার বয়স ৩০ বছর। থাইরয়েড সমস্যা আছে, ওজন বেড়ে যাচ্ছে এবং ক্লান্ত লাগে। Endocrinologist TSH test করতে বলেছেন এবং Levothyroxine দিয়েছেন।\"\n"
        "Answer: {\"Age\":[\"৩০ বছর\"], "
        "\"Symptom\":[\"ওজন বেড়ে যাচ্ছে\",\"ক্লান্ত লাগে\"], "
        "\"Medicine\":[\"Levothyroxine\"], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\"], "
        "\"Specialist\":[\"Endocrinologist\"], "
        "\"Medical_Procedure\":[\"TSH test\"]}\n\n"
    ),
]


FEWSHOT_01: List[str] = [
    (
        "Text: \"আমার পিত্ত থলিতে পাথর আছে। গত ২.৫ বছর যাবৎ এটা হয়েছে। যার আকার ৪ সেমি। এতদিন তীব্র কোন ব্যাথা ছিল না কিন্তু গত ৮ দিনের মধ্যে ৩ দিন পেটে তীব্র ব্যাথা হয়েছিল এবং পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়। পিত্ত থলিতে পাথর হলে কী পেট ব্যাথার সাথে সাথে পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়? এবং কেন এই চাপ দিলে ব্যাথা অনুভূত হয়?? অভিজ্ঞ ডাক্তারের পরামর্শ চাচ্ছি।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"ব্যাথা\",\"পেটে তীব্র ব্যাথা\",\"চাপ দিলে ব্যাথা\",\"পেট ব্যাথার\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পিত্ত থলিতে পাথর\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"আস্সালামুয়ালাইকুম, আমার বয়স 22 বছর. গত দেড় মাস যাবত আমার পালস অনেক বেশি থাকে ( বেশি সময় 100 এর উপরে ) বিশেষ করে খাবার খাওয়ার পর. সাথে দুর্বলতা থাকায় এক মাস আগে আমি একজন মেডিসিন বিশেষজ্ঞ কে দেখাই. তিনি ECG এবং ECHO করাতে বলেন কিন্তু রিপোর্টে তেমন অস্বাভাবিক কিছু না থাকায় তিনি আমাকে PROPRANOLOL HCL 10 mg ৩০ দিন এবং ALPRAZOLAM 0. 25 mg ১০ দিন খেতে বলেন. এখন ঔষধ শেষ হওয়ার পর এ সমস্যা যায়নি. বরং ঔষধ খাওয়ার কিছুদিন পর থেকে আমার শ্বাস নিতে কষ্ট হচ্ছে. এর আগে কখনো শ্বাসকষ্ট হয়নি. আমার জ্বর নেই, এখন এই শ্বাস নিতে কষ্ট এবং অতিরিক্ত পালস এর কারনে স্বাভাবিক থাকতে পারছি না আর এমন সংকটময় পরিস্থিতি তে ডাক্তার এর কাছেও যেতে পারছি না. উল্লেখ্য: আমার ঠান্ডার সমস্যা থাকায় অনেক দিন আগে থেকেই Fexofenadine Hydrochloride 120 mg খাই।\"\n"
        "Answer: {\"Age\":[\"22 বছর\"], "
        "\"Symptom\":[\"পালস অনেক বেশি\",\"দুর্বলতা\",\"শ্বাস নিতে কষ্ট\",\"শ্বাসকষ্ট\",\"জ্বর\",\"অতিরিক্ত পালস\",\"ঠান্ডার সমস্যা\"], "
        "\"Medicine\":[\"PROPRANOLOL HCL 10 mg\",\"ALPRAZOLAM 0\",\"Fexofenadine Hydrochloride 120 mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"ECG\",\"ECHO\"]}\n\n"
    ),
    (
        "Text: \"আমার বাবা, মা ভাই, বোন কারো এ্যাজমা নেই। আমার এলাজি আছে, বুক ভারি হয়, শ্বাস নিতে সমস্যা হয়। ইকো, ইসিজি, চেস্ট এক্সরে রিপোর্ট নরমাল। আমি এই সমস্যার জন্য কোন ডাক্তার দেখাবো? ডাক্তারভাই এ কি এই বিষয়ে অভিজ্ঞ ডাক্তার আছেন?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"বুক ভারি হয়\",\"শ্বাস নিতে সমস্যা\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"এ্যাজমা\",\"এলাজি\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[\"ইকো\",\"ইসিজি\",\"চেস্ট এক্সরে\"]}\n\n"
    ),
    (
        "Text: \"she had pain in her left back shoulder then had rolac 10 days then visited a government hospital a doctor prescribed these: Flexi 100mg, Flexllax 10mg, Cosec 20mg, Neurolin 25. Still no change. What specialist doctor should she visit?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"pain in her left back\"], "
        "\"Medicine\":[\"rolac\",\"Flexi\",\"Flexllax\",\"Cosec\",\"Neurolin\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"আসসালামু আলাইকুম স্যার, স্যার আমার আম্মুর অস্থিরতা ও শাসকষ্ট, এতে আবার পায়েও পানি আসে। এতে কি হৃদ রোগ হইছে? আর কি করবো বুঝতে পারছি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"অস্থিরতা\",\"শাসকষ্ট\",\"পায়েও পানি আসে\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"হৃদ রোগ\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"এলার্জির সমস্যার কারণে রোদে গেলে গা চিটমিট করে, মাথার ভিতরে কিলবিল করে। সকালে ঘুম থেকে উঠলে অনবরত হাঁচি হয়। কখনো নিয়মিত কোনো এলার্জির ওষুধ খাইনি। এক্ষেত্রে আমি কি করতে পারি? এনার্জির কারণে অনেক দৈনন্দিন কাজ করতে পারি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"রোদে গেলে গা চিটমিট করে\",\"মাথার ভিতরে কিলবিল করে\",\"হাঁচি\"], "
        "\"Medicine\":[\"এলার্জির ওষুধ\"], "
        "\"Health_Condition\":[\"এলার্জির\",\"এনার্জির\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"গত সোমবার হৈতে আমার হাড়ের জয়েন্টে ব্যথা, গা ম্যাচ ম্যাচ করে। ভিতরে ভিতরে জ্বর লাগে কিন্তু মাপলে ৯৯। নাপা এক্সটা খাইতেছি। উল্লেখ্য ৩/৪ আগে ডিক্স পোলাপ্স ছিল। শূকনো কাশি নাই, তবে টুটি গিলতে হালকা ব্যাথা লাগে। করোনা ভয়ে ivermactin 12mg 2pc ও ডক সিন100 - ৫টি খাইছি। জ্বর সব সময় থাকে না, শরির খুব দুর্লভ।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"হাড়ের জয়েন্টে ব্যথা\",\"গা ম্যাচ ম্যাচ করে\",\"ভিতরে ভিতরে জ্বর লাগে\",\"টুটি গিলতে হালকা ব্যাথা লাগে\",\"জ্বর\",\"শরির খুব দুর্লভ\"], "
        "\"Medicine\":[\"নাপা এক্সটা\",\"ivermactin 12mg\",\"ডক সিন100\"], "
        "\"Health_Condition\":[\"ডিক্স পোলাপ্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"গত সপ্তাহ থেকে জ্বর আছে, রাতে কাশি বাড়ে, মাঝে মাঝে বমি হয়.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"জ্বর\",\"কাশি\",\"বমি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"বুকে ব্যথা হচ্ছে এবং শ্বাসকষ্ট আছে; খুব অস্বস্তি লাগছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\",\"অস্বস্তি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"রোগী অ্যাজমা ও গ্যাস্ট্রাইটিসের রোগী; গতকাল ধরা পড়েছে মাইগ্রেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"অ্যাজমা\",\"গ্যাস্ট্রাইটিস\",\"মাইগ্রেন\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"থাইরয়েড সমস্যা আছে; history of ডায়াবেটিস উল্লেখ আছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\",\"ডায়াবেটিস\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"গত চার মাস ধরে মাথা ব্যথা; কাশি না.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"মাথা ব্যথা\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"Your serum Triglyceride is slightly raised; HbA1c 6.5%.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"গত তিন দিন ধরে কাশি ও জ্বর আছে। একজন মেডিসিন বিশেষজ্ঞ আমাকে সেফিক্সিম দিয়েছেন। এক্স-রে করা হয়নি.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"কাশি\",\"জ্বর\"], \"Medicine\":[\"সেফিক্সিম\"], "
        "\"Health_Condition\":[], \"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"রোগীর বয়স ৫৫ বছর। তিনি ডায়াবেটিস ও উচ্চ রক্তচাপের রোগী এবং মেটফরমিন ও লোসারটান খাচ্ছেন.\"\n"
        "Answer: {\"Age\":[\"৫৫ বছর\"], \"Symptom\":[], \"Medicine\":[\"মেটফরমিন\",\"লোসারটান\"], "
        "\"Health_Condition\":[\"ডায়াবেটিস\",\"উচ্চ রক্তচাপ\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"ইএনটি বিশেষজ্ঞ টিম্পানোমেট্রি করতে বলেছেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[\"ইএনটি বিশেষজ্ঞ\"], \"Medical_Procedure\":[\"টিম্পানোমেট্রি\"]}\n\n"
    ),
    (
        "Text: \"আমার বাচ্চার ৫০ দিন। দুই তিন দিন যাবত শুকনা ঠান্ডা মনে হচ্ছে। নাক বন্ধ হয়ে থাকে, শ্বাস নিতে কষ্ট হয়। আমি নজোমিষ্ট বিপি০. ৯ % ব্যবহার করছি কিন্তু নাক খোলছে না। এখন আর কি কোন ঔষধ খাওয়াতে হবে আর কি ঔষধ খাওয়াতে পারি??\"\n"
        "Answer: {\"Age\":[\"৫০ দিন\"], \"Symptom\":[\"শুকনা ঠান্ডা\",\"নাক বন্ধ\",\"শ্বাস নিতে কষ্ট\",\"নাক খোলছে না\"], "
        "\"Medicine\":[\"নজোমিষ্ট বিপি০\"], \"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Text: \"আমার ছেলের বয়স ১০ বছর। ওর গায়ে পক্স উঠেছিল। এখন ভাল হয়ে গেছে। কিন্তু শরীরে সাদা সাদা দাগ আবার কিছু কাল কাল দাগ দেখা যাচ্ছে। এগুলো কি এমনিতেই চলে যাবে নাকি কোন ওষুধ ব্যবহার করতে হবে?\"\n"
        "Answer: {\"Age\":[\"১০ বছর\"], "
        "\"Symptom\":[\"শরীরে সাদা সাদা দাগ\",\"কাল কাল দাগ\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পক্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
]


QA_FEWSHOTS_ALL: List[str] = [
    (
        "Description of text: The patient says they have stones in the gallbladder for about 2.5 years, the stone is 4 cm, and recently they had several episodes of severe abdominal pain with tenderness under the right ribs. They are asking if this pain pattern is typical for gallbladder stones and want an experienced doctor’s advice. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in text.\n"
        "Text: \"আমার পিত্ত থলিতে পাথর আছে। গত ২.৫ বছর যাবৎ এটা হয়েছে। যার আকার ৪ সেমি। এতদিন তীব্র কোন ব্যাথা ছিল না কিন্তু গত ৮ দিনের মধ্যে ৩ দিন পেটে তীব্র ব্যাথা হয়েছিল এবং পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়। পিত্ত থলিতে পাথর হলে কী পেট ব্যাথার সাথে সাথে পেটের ডানদিকে পাজরের নিচে চাপ দিলে ব্যাথা অনুভূত হয়? এবং কেন এই চাপ দিলে ব্যাথা অনুভূত হয়?? অভিজ্ঞ ডাক্তারের পরামর্শ চাচ্ছি।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"ব্যাথা\",\"পেটে তীব্র ব্যাথা\",\"চাপ দিলে ব্যাথা\",\"পেট ব্যাথার\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পিত্ত থলিতে পাথর\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: A 22-year-old patient reports persistently high pulse, especially after meals, plus weakness. A medicine specialist advised ECG and ECHO, which were mostly normal, and prescribed several medicines, but the problem and new shortness of breath continue. The patient also has a history of cold problems and takes Fexofenadine regularly. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in text.\n"
        "Text: \"আস্সালামুয়ালাইকুম, আমার বয়স 22 বছর. গত দেড় মাস যাবত আমার পালস অনেক বেশি থাকে ( বেশি সময় 100 এর উপরে ) বিশেষ করে খাবার খাওয়ার পর. সাথে দুর্বলতা থাকায় এক মাস আগে আমি একজন মেডিসিন বিশেষজ্ঞ কে দেখাই. তিনি ECG এবং ECHO করাতে বলেন কিন্তু রিপোর্টে তেমন অস্বাভাবিক কিছু না থাকায় তিনি আমাকে PROPRANOLOL HCL 10 mg ৩০ দিন এবং ALPRAZOLAM 0. 25 mg ১০ দিন খেতে বলেন. এখন ঔষধ শেষ হওয়ার পর এ সমস্যা যায়নি. বরং ঔষধ খাওয়ার কিছুদিন পর থেকে আমার শ্বাস নিতে কষ্ট হচ্ছে. এর আগে কখনো শ্বাসকষ্ট হয়নি. আমার জ্বর নেই, এখন এই শ্বাস নিতে কষ্ট এবং অতিরিক্ত পালস এর কারনে স্বাভাবিক থাকতে পারছি না আর এমন সংকটময় পরিস্থিতি তে ডাক্তার এর কাছেও যেতে পারছি না. উল্লেখ্য: আমার ঠান্ডার সমস্যা থাকায় অনেক দিন আগে থেকেই Fexofenadine Hydrochloride 120 mg খাই.\"\n"
        "Answer: {\"Age\":[\"22 বছর\"], "
        "\"Symptom\":[\"পালস অনেক বেশি\",\"দুর্বলতা\",\"শ্বাস নিতে কষ্ট\",\"শ্বাসকষ্ট\",\"জ্বর\",\"অতিরিক্ত পালস\",\"ঠান্ডার সমস্যা\"], "
        "\"Medicine\":[\"PROPRANOLOL HCL 10 mg\",\"ALPRAZOLAM 0\",\"Fexofenadine Hydrochloride 120 mg\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], "
        "\"Medical_Procedure\":[\"ECG\",\"ECHO\"]}\n\n"
    ),
    (
        "Description of text: The writer explains that none of their close family members have asthma, but they themselves have allergy problems, feel chest heaviness, and have trouble breathing. Echo, ECG, and chest X-ray reports are normal. They are asking which doctor to see and if there is an experienced doctor for this issue. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in text.\n"
        "Text: \"আমার বাবা, মা ভাই, বোন কারো এ্যাজমা নেই। আমার এলাজি আছে, বুক ভারি হয়, শ্বাস নিতে সমস্যা হয়। ইকো, ইসিজি, চেস্ট এক্সরে রিপোর্ট নরমাল। আমি এই সমস্যার জন্য কোন ডাক্তার দেখাবো? ডাক্তারভাই এ কি এই বিষয়ে অভিজ্ঞ ডাক্তার আছেন?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"বুক ভারি হয়\",\"শ্বাস নিতে সমস্যা\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"এ্যাজমা\",\"এলাজি\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[\"ইকো\",\"ইসিজি\",\"চেস্ট এক্সরে\"]}\n\n"

    ),
    (
        "Description of text: The patient (or caregiver) says she has pain in her left back and shoulder, took Rolac for 10 days, then went to a government hospital where several medicines were prescribed. There is still no improvement, and they are asking which specialist doctor she should visit. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in text.\n"
        "Text: \"she had pain in her left back shoulder then had rolac 10 days then visited a government hospital a doctor prescribed these: Flexi 100mg, Flexllax 10mg, Cosec 20mg, Neurolin 25. Still no change. What specialist doctor should she visit?\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"pain in her left back\"], "
        "\"Medicine\":[\"rolac\",\"Flexi\",\"Flexllax\",\"Cosec\",\"Neurolin\"], "
        "\"Health_Condition\":[], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The writer describes their mother’s restlessness, shortness of breath, and swelling in the legs. They are worried that this may be heart disease and are unsure what to do next. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in text.\n"
        "Text: \"আসসালামু আলাইকুম স্যার, স্যার আমার আম্মুর অস্থিরতা ও শাসকষ্ট, এতে আবার পায়েও পানি আসে। এতে কি হৃদ রোগ হইছে? আর কি করবো বুঝতে পারছি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"অস্থিরতা\",\"শাসকষ্ট\",\"পায়েও পানি আসে\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"হৃদ রোগ\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The patient has allergy problems; going in the sun causes skin discomfort, there is a crawling sensation in the head, and they sneeze continuously in the morning. They have not taken allergy medicine regularly and are asking what they can do because the allergy interferes with daily activities. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in text.\n"
        "Text: \"এলার্জির সমস্যার কারণে রোদে গেলে গা চিটমিট করে, মাথার ভিতরে কিলবিল করে। সকালে ঘুম থেকে উঠলে অনবরত হাঁচি হয়। কখনো নিয়মিত কোনো এলার্জির ওষুধ খাইনি। এক্ষেত্রে আমি কি করতে পারি? এনার্জির কারণে অনেক দৈনন্দিন কাজ করতে পারি না।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"রোদে গেলে গা চিটমিট করে\",\"মাথার ভিতরে কিলবিল করে\",\"হাঁচি\"], "
        "\"Medicine\":[\"এলার্জির ওষুধ\"], "
        "\"Health_Condition\":[\"এলার্জির\",\"এনার্জির\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: Since last Monday, the patient has pain in the joints, body aches, and a feeling of internal fever around 99°F. They are taking Napa Extra and mention a history of disc prolapse 3–4 years ago. There is no dry cough, but there is mild throat pain when swallowing, and they took ivermectin and another medicine out of fear of COVID. The fever is not constant, and the body feels very weak. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text.\n"
        "Text: \"গত সোমবার হৈতে আমার হাড়ের জয়েন্টে ব্যথা, গা ম্যাচ ম্যাচ করে। ভিতরে ভিতরে জ্বর লাগে কিন্তু মাপলে ৯৯। নাপা এক্সটা খাইতেছি। উল্লেখ্য ৩/৪ আগে ডিক্স পোলাপ্স ছিল। শূকনো কাশি নাই, তবে টুটি গিলতে হালকা ব্যাথা লাগে। করোনা ভয়ে ivermactin 12mg 2pc ও ডক সিন100 - ৫টি খাইছি। জ্বর সব সময় থাকে না, শরির খুব দুর্লভ।\"\n"
        "Answer: {\"Age\":[], "
        "\"Symptom\":[\"হাড়ের জয়েন্টে ব্যথা\",\"গা ম্যাচ ম্যাচ করে\",\"ভিতরে ভিতরে জ্বর লাগে\",\"টুটি গিলতে হালকা ব্যাথা লাগে\",\"জ্বর\",\"শরির খুব দুর্লভ\"], "
        "\"Medicine\":[\"নাপা এক্সটা\",\"ivermactin 12mg\",\"ডক সিন100\"], "
        "\"Health_Condition\":[\"ডিক্স পোলাপ্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The patient reports that they have had fever for the last week, their cough gets worse at night, and they occasionally vomit. Your task is to identify which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"গত সপ্তাহ থেকে জ্বর আছে, রাতে কাশি বাড়ে, মাঝে মাঝে বমি হয়.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"জ্বর\",\"কাশি\",\"বমি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The patient reports having chest pain, shortness of breath, and a strong feeling of discomfort. These symptoms suggest an urgent physical issue. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "are present in this text?\n"
        "Text: \"বুকে ব্যথা হচ্ছে এবং শ্বাসকষ্ট আছে; খুব অস্বস্তি লাগছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\",\"অস্বস্তি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The text states that the patient has asthma and gastritis as existing conditions, and was newly diagnosed with migraine yesterday. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"রোগী অ্যাজমা ও গ্যাস্ট্রাইটিসের রোগী; গতকাল ধরা পড়েছে মাইগ্রেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"অ্যাজমা\",\"গ্যাস্ট্রাইটিস\",\"মাইগ্রেন\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The text states that the patient has a thyroid problem and also mentions a history of diabetes. From reading this information, you must identify which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"থাইরয়েড সমস্যা আছে; history of ডায়াবেটিস উল্লেখ আছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\",\"ডায়াবেটিস\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The patient reports having headaches continuously for the last four months, but they do not have any cough. From this text, you have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"গত চার মাস ধরে মাথা ব্যথা; কাশি না.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"মাথা ব্যথা\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: This text reports two laboratory findings — slightly elevated serum triglycerides and an HbA1c of 6.5%. There are no symptoms, medications, diagnoses, specialists, or procedures directly mentioned. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"Your serum Triglyceride is slightly raised; HbA1c 6.5%.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
        "are present in this text?\n"
        "Text: \"গত তিন দিন ধরে কাশি ও জ্বর আছে। একজন মেডিসিন বিশেষজ্ঞ আমাকে সেফিক্সিম দিয়েছেন। এক্স-রে করা হয়নি.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"কাশি\",\"জ্বর\"], \"Medicine\":[\"সেফিক্সিম\"], "
        "\"Health_Condition\":[], \"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The text describes a 55-year-old patient who has diabetes and high blood pressure, and is currently taking Metformin and Losartan. It states the patient’s age, chronic health conditions, and the medicines being used. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "are present in this text?\n"
        "Text: \"রোগীর বয়স ৫৫ বছর। তিনি ডায়াবেটিস ও উচ্চ রক্তচাপের রোগী এবং মেটফরমিন ও লোসারটান খাচ্ছেন.\"\n"
        "Answer: {\"Age\":[\"৫৫ বছর\"], \"Symptom\":[], \"Medicine\":[\"মেটফরমিন\",\"লোসারটান\"], "
        "\"Health_Condition\":[\"ডায়াবেটিস\",\"উচ্চ রক্তচাপ\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The text states that an ENT specialist advised the patient to undergo a tympanometry test. From this information, you have to identify which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"ইএনটি বিশেষজ্ঞ টিম্পানোমেট্রি করতে বলেছেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[\"ইএনটি বিশেষজ্ঞ\"], \"Medical_Procedure\":[\"টিম্পানোমেট্রি\"]}\n\n"
    ),
    (
        "Description of text: The parent describes that their baby is 50 days old and for the last two to three days has symptoms such as dry cold, blocked nose, and difficulty breathing. They are using Nazomist BP 0.9% nasal drops, but the nose is still not opening. They want to know if more medicine is needed and what medicines can be given. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"আমার বাচ্চার ৫০ দিন। দুই তিন দিন যাবত শুকনা ঠান্ডা মনে হচ্ছে। নাক বন্ধ হয়ে থাকে, শ্বাস নিতে কষ্ট হয়। আমি নজোমিষ্ট বিপি০. ৯ % ব্যবহার করছি কিন্তু নাক খোলছে না। এখন আর কি কোন ঔষধ খাওয়াতে হবে আর কি ঔষধ খাওয়াতে পারি??\"\n"
        "Answer: {\"Age\":[\"৫০ দিন\"], \"Symptom\":[\"শুকনা ঠান্ডা\",\"নাক বন্ধ\",\"শ্বাস নিতে কষ্ট\",\"নাক খোলছে না\"], "
        "\"Medicine\":[\"নজোমিষ্ট বিপি০\"], \"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Description of text: The parent explains that their 10-year-old child recently had pox, which has now resolved. However, white and dark marks remain on the skin. They are asking whether these spots will fade naturally or if any medicine is needed. You have to find out which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in the text.\n"
        "Text: \"আমার ছেলের বয়স ১০ বছর। ওর গায়ে পক্স উঠেছিল। এখন ভাল হয়ে গেছে। কিন্তু শরীরে সাদা সাদা দাগ আবার কিছু কাল কাল দাগ দেখা যাচ্ছে। এগুলো কি এমনিতেই চলে যাবে নাকি কোন ওষুধ ব্যবহার করতে হবে?\"\n"
        "Answer: {\"Age\":[\"১০ বছর\"], "
        "\"Symptom\":[\"শরীরে সাদা সাদা দাগ\",\"কাল কাল দাগ\"], "
        "\"Medicine\":[], "
        "\"Health_Condition\":[\"পক্স\"], "
        "\"Specialist\":[], "
        "\"Medical_Procedure\":[]}\n\n"
    ),  
]

# Base QA-style user template (English)
USER_TEMPLATE_QA_EN = (
    "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) "
    "are present in this text?\n"
    "Text:\n<<<\n{input_text}\n>>>\n"
    "Answer (a single JSON object with those six keys):\n"
)

# Bangla-wrapped QA template
USER_TEMPLATE_QA_BN = (
    "প্রশ্ন: এই টেক্সটে কোন কোন entity (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) আছে?\n"
    "টেক্সট:\n<<<\n{input_text}\n>>>\n"
    "উত্তর (উপরের ছয়টি key সহ একটিমাত্র JSON অবজেক্ট):\n"
)


USER_TEMPLATE_PLAIN_EN = (
    "Text:\n<<<\n{input_text}\n>>>\n"
    "Extract all entities for the schema:\n"
    "Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure.\n"
    "Return only a single JSON object with exactly these keys:\n"
)

USER_TEMPLATE_ZERO_SHOT = (
    "Text:\n<<<\n{input_text}\n>>>\n"
)

@dataclass
class PromptConfig:
    system_base: str
    fewshots: List[str]
    user_template: str


def _make_bn_fewshots(base_fewshots: List[str]) -> List[str]:
    """Convert 'Question/Text/Answer' labels in few-shots to Bangla wrappers."""
    converted = []
    for ex in base_fewshots:
        ex_bn = (
            ex.replace("Question:", "প্রশ্ন:")
              .replace("Text:", "টেক্সট:")
              .replace("Answer:", "উত্তর:")
        )
        converted.append(ex_bn)
    return converted


def get_prompt_config(variant: str) -> PromptConfig:
    """
    Return the prompt configuration for a given variant.

    Supported variants:
      - 'en_bn_Description"'   : Explanation based prompting
      - 'qa_en'   : translation-based propting
      - 'en_bn_QA'   : question-based propting
      - 'plain_en': few shots prompting
      - 'zero_shot': zero-shot prompting
    """
    if variant == "en_bn_Description": #5. explanation
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_EN,
            fewshots=_make_bn_fewshots(QA_FEWSHOTS_ALL),
            user_template=USER_TEMPLATE_PLAIN_EN, #USER_TEMPLATE_QA_BN
        )
    elif variant == "qa_bn":
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_BN,
            fewshots=_make_bn_fewshots(BASE_FEWSHOTS),
            user_template=USER_TEMPLATE_QA_BN,
        )
    elif variant == "qa_en":    #4. translation-based
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_BN,
            fewshots=_make_bn_fewshots(Translation_based_FEWSHOTS_01),
            user_template=USER_TEMPLATE_PLAIN_EN,
        )
    elif variant == "plain_en":   #2. few shot
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_EN,
            fewshots=FEWSHOT_01,  # examples still fine; only wrapper style changes
            user_template=USER_TEMPLATE_PLAIN_EN,
        )
    elif variant == "en_bn_QA":  #3. Question-based 
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_EN,   # best result by SYSTEM_PROMPT_BASE_EN,
            fewshots=_make_bn_fewshots(QA_FEWSHOTS_01),  #best QA_FEWSHOTS_01
            user_template=USER_TEMPLATE_QA_BN,  # best USER_TEMPLATE_QA_BN
        )
    elif variant == "zero_shot":  #1. zero shot
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_EN,  #SYSTEM_PROMPT_BASE_EN
            fewshots=[],  
            user_template=USER_TEMPLATE_ZERO_SHOT ,
        )
    else:
        # Fallback: default to qa_en
        return PromptConfig(
            system_base=SYSTEM_PROMPT_BASE_EN,
            fewshots=BASE_FEWSHOTS,
            user_template=USER_TEMPLATE_QA_EN,
        )


# ---------- JSON helpers ----------
def extract_first_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def normalize_json_obj(obj: dict) -> dict:
    fixed = {}
    for k in SCHEMA_KEYS:
        vals = obj.get(k, [])
        if not isinstance(vals, list):
            vals = []
        norm = []
        seen = set()
        for v in vals:
            if isinstance(v, str):
                vv = " ".join(v.split()).strip()
                if vv and vv not in seen:
                    seen.add(vv)
                    norm.append(vv)
        fixed[k] = norm
    return fixed


def decode_to_json(text: str) -> dict:
    js = extract_first_json(text)
    if not js:
        return {k: [] for k in SCHEMA_KEYS}
    try:
        obj = json.loads(js)
    except Exception:
        js2 = (js.replace("None", "null")
               .replace("True", "true")
               .replace("False", "false"))
        try:
            obj = json.loads(js2)
        except Exception:
            return {k: [] for k in SCHEMA_KEYS}
    return normalize_json_obj(obj)


def load_jsonl(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            yield json.loads(s)


def save_jsonl(path: str, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- Token budgeting helpers ----------
def count_tokens(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def fit_system_with_examples(
    llm: Llama,
    ctx: int,
    max_gen_tokens: int,
    cfg: PromptConfig,
) -> str:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    base = cfg.system_base
    tokens = count_tokens(llm, base)
    pieces: List[str] = []
    used = 0
    for ex in cfg.fewshots:
        ex_tokens = count_tokens(llm, ex)
        if tokens + ex_tokens > budget:
            break
        pieces.append(ex)
        tokens += ex_tokens
        used += 1
    print(f"[DEBUG] Using {used} few-shot examples for this variant.")
    return base + "".join(pieces)


def maybe_truncate_input(
    llm: Llama,
    input_text: str,
    ctx: int,
    sys_str: str,
    max_gen_tokens: int,
    user_template: str,
) -> Tuple[str, int, int]:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    user_block = user_template.format(input_text=input_text)
    total = count_tokens(llm, sys_str) + count_tokens(llm, user_block)
    if total <= budget:
        return input_text, total, budget

    toks_input = llm.tokenize(input_text.encode("utf-8"), add_bos=False)
    # keep some tail tokens from input
    keep = max(64, budget - count_tokens(llm, sys_str) - count_tokens(llm, user_template.format(input_text="")))
    if keep < 64:
        keep = 64
    truncated_ids = toks_input[-keep:]
    truncated_text = llm.detokenize(truncated_ids).decode("utf-8", errors="ignore")
    user_block2 = user_template.format(input_text=truncated_text)
    total2 = count_tokens(llm, sys_str) + count_tokens(llm, user_block2)
    return truncated_text, total2, budget


def build_messages(
    llm: Llama,
    input_text: str,
    ctx: int,
    max_gen_tokens: int,
    cfg: PromptConfig,
) -> List[Dict[str, str]]:
    system_str = fit_system_with_examples(llm, ctx, max_gen_tokens, cfg)
    input_text_fitted, _, _ = maybe_truncate_input(
        llm,
        input_text,
        ctx,
        system_str,
        max_gen_tokens,
        cfg.user_template,
    )
    return [
        {"role": "system", "content": system_str},
        {"role": "user", "content": cfg.user_template.format(input_text=input_text_fitted.strip())},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    help="Path to GGUF model file")
    ap.add_argument("--input", default="data/data_llm_io.jsonl")
    ap.add_argument("--out", default="data/preds_llama31_8b_ex_promptvar.jsonl")
    ap.add_argument("--ctx", type=int, default=4096, help="Context window")
    ap.add_argument("--n_gpu_layers", type=int, default=0,
                    help="Set >0 (or -1) to offload layers to GPU if built with CUDA/Metal/OpenCL")
    ap.add_argument("--n_threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="Slight recall nudge; 0.0 for max determinism")
    ap.add_argument("--top_p", type=float, default=0.9,
                    help="Slight recall nudge; 1.0 for max determinism")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--prompt_variant",
        default="qa_en",
        choices=["qa_en", "qa_bn", "plain_en", "en_bn_QA", "en_bn_Description",
        "zero_shot"],
        help=(
            "Which prompt style to use:\n"
            "  qa_en   = QA-style, English instructions (original style)\n"
            "  qa_bn   = QA-style, Bangla instructions\n"
            "  en_bn_Description = Description-style, English Bangla\n"
            "  plain_en = Instruction-style, English (no explicit Question:)\n"
            "  en_bn_QA = QA-style, English Bangla instructions\n"
        ),
    )

    args = ap.parse_args()

    print(f"Loading GGUF: {args.model_path}")
    print(f"Using prompt variant: {args.prompt_variant}")

    cfg = get_prompt_config(args.prompt_variant)

    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        seed=args.seed,
        chat_format="llama-3",   # LLaMA-3 / 3.1 instruct
        verbose=False,
    )

    records = list(load_jsonl(args.input))
    out_rows = []

    for r in tqdm(records, desc="Generating"):
        messages = build_messages(llm, r["Input"], args.ctx, args.max_tokens, cfg)
        out = llm.create_chat_completion(
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        content = out["choices"][0]["message"]["content"]
        pred = decode_to_json(content)
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})

    save_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
