"""
Generate a synthetic exam anxiety dataset for training.
Produces labeled text samples across Low, Moderate, and High anxiety categories.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DATASET_PATH

# ── Templates per anxiety level ──────────────────────────────────────────────

LOW_TEMPLATES = [
    "I feel pretty confident about the upcoming exam. I've studied well.",
    "I'm looking forward to the exam. I've prepared thoroughly.",
    "I have a solid study plan and feel ready for the test.",
    "I feel calm and prepared for the examination tomorrow.",
    "I've reviewed all the material and feel good about my preparation.",
    "The exam doesn't worry me much. I understand the topics well.",
    "I feel relaxed about the test. My notes are organized and I've practiced.",
    "I'm confident I can handle the exam questions based on my preparation.",
    "Exams are a normal part of learning. I feel ready for this one.",
    "I studied consistently throughout the semester so I'm not stressed.",
    "I feel well-prepared. I reviewed past papers and practiced regularly.",
    "The material makes sense to me. I think the exam will go smoothly.",
    "I organized my study schedule well and covered all major topics.",
    "I'm not worried about the exam. I've done my best in preparation.",
    "I feel positive about taking this exam. My hard work will pay off.",
    "I had a good study group that helped me understand difficult concepts.",
    "I feel balanced and focused. Studying was manageable this semester.",
    "The practice tests went well, so I'm optimistic about the real exam.",
    "I enjoy this subject, so studying felt natural and productive.",
    "I feel collected and stable. I know what to expect in the exam.",
    "My preparation has been consistent. I feel strong going in.",
    "I completed all assignments and attended every class. I feel ready.",
    "I slept well last night and feel refreshed for studying.",
    "I believe in my abilities and my study efforts this semester.",
    "I practiced time management for the exam and feel comfortable.",
    "I feel grateful for the resources I had to prepare properly.",
    "This subject clicks for me. I don't anticipate major problems.",
    "I feel content with how much I studied. I gave it my all.",
    "I've studied similar material before, so this feels familiar.",
    "I feel centered. I'm ready to demonstrate what I've learned.",
    "I discussed hard questions with friends and cleared my doubts.",
    "I attended review sessions and feel confident in my comprehension.",
    "I used flashcards and practice problems to reinforce my knowledge.",
    "I'm at peace with my preparation level and confident overall.",
    "I feel equipped to answer most questions on this exam.",
    "I feel comfortable because I started preparing early this time.",
    "I managed my time well this semester and feel low stress.",
    "I feel happy and ready. I enjoyed the learning process.",
    "I have no significant concerns about the exam content.",
    "I feel light-hearted about exam week. I've put in solid effort.",
]

MODERATE_TEMPLATES = [
    "I've studied but still feel a bit nervous about some topics.",
    "I'm somewhat anxious about the exam. Some chapters are confusing.",
    "I hope I remember everything during the test. I'm a bit worried.",
    "There are a few topics I'm not sure about. I feel uneasy.",
    "I studied hard but I'm still worried about the essay questions.",
    "I feel nervous but I think I can manage if I stay focused.",
    "Some parts of the syllabus are tricky. I'm slightly stressed.",
    "I have mixed feelings about the exam. Some topics I know well, others not.",
    "I'm a little anxious because the exam format has changed this year.",
    "I wish I had more time to prepare. I feel moderately stressed.",
    "I get nervous during exams even when I'm prepared. I hope I stay calm.",
    "I've studied the main topics but I'm worried about surprise questions.",
    "I feel somewhat pressured by the time limit on the exam.",
    "I'm okay with most subjects but math problems make me uneasy.",
    "I studied with friends, but I still have some doubts about a few topics.",
    "The exam is important for my grade and that adds some pressure.",
    "I sometimes blank out during tests and that worries me a little.",
    "I feel prepared but the stakes are high, which makes me tense.",
    "I have butterflies in my stomach thinking about the exam.",
    "I studied well, but last-minute doubts keep creeping in.",
    "I wish I had practiced more problems. I feel a bit underprepared.",
    "I keep second-guessing my answers when I study. It makes me anxious.",
    "I feel okay about theoretical questions but worried about practical ones.",
    "The competitive atmosphere among students adds to my stress.",
    "I'm somewhat worried because I didn't do well on the midterm.",
    "I feel tense when I think about how much material there is to cover.",
    "I can't tell if I studied enough. I have a nagging sense of worry.",
    "I'm slightly anxious about time management during the actual exam.",
    "I feel a knot in my stomach but I think I can push through.",
    "I feel uncertain about a couple of chapters but okay about the rest.",
    "I'm a bit nervous about the exam but trying to stay positive.",
    "I feel restless the night before exams. I hope I can sleep well.",
    "I studied but I keep worrying about forgetting important formulas.",
    "I feel mild tension because this exam impacts my overall grade.",
    "Sometimes my mind goes blank during tests and that scares me a bit.",
    "I feel somewhat apprehensive but I believe I can manage.",
    "I feel nervous when I compare my preparation with top students.",
    "I'm worried about multiple-choice questions that try to trick you.",
    "I feel a moderate amount of stress but I'm trying to cope.",
    "I prepared but feel unsure about how confident I really am.",
]

HIGH_TEMPLATES = [
    "I can't stop thinking about the exam. I feel overwhelmed and scared.",
    "I haven't been able to sleep because of exam stress. I feel terrible.",
    "I feel like I'm going to fail no matter how much I study.",
    "My heart races every time I think about the exam. I feel paralyzed.",
    "I'm extremely anxious. I can't concentrate and keep crying.",
    "I feel completely unprepared and hopeless about the exam.",
    "The thought of the exam makes me physically ill. I feel nauseous.",
    "I've been having panic attacks because of the upcoming exam.",
    "I can't eat or sleep properly. The exam pressure is destroying me.",
    "I feel like dropping out. The exam stress is unbearable.",
    "I'm terrified of failing. My parents will be so disappointed.",
    "I feel so much pressure that I can't even open my textbooks.",
    "Every time I try to study, I break down crying from the stress.",
    "I feel like my entire future depends on this one exam and it's crushing.",
    "I'm having nightmares about failing the exam. I feel desperate.",
    "The anxiety is so bad I can't function normally. I need help.",
    "I feel completely overwhelmed. There's too much material and too little time.",
    "I've been shaking and sweating just thinking about the exam.",
    "I feel trapped and suffocated by the exam pressure.",
    "I can't breathe properly when I think about the test. It's crippling.",
    "I feel worthless because I can't keep up with the study schedule.",
    "I'm not eating. I'm not sleeping. All I do is panic about exams.",
    "I feel a constant knot in my chest that won't go away.",
    "I dread walking into the exam hall. I feel frozen with fear.",
    "My hands shake when I try to write during exams. I can't control it.",
    "I feel isolated because everyone else seems calm and I'm falling apart.",
    "I keep imagining the worst outcomes and I can't stop the spiral.",
    "I feel so stressed that I've started getting headaches and stomach aches.",
    "I've lost all motivation. The anxiety has taken over completely.",
    "I feel like I'm drowning in stress and there's no way out.",
    "My heart pounds so hard I can hear it. The fear is consuming me.",
    "I feel completely burnt out. The exam pressure has been relentless.",
    "I cry every night because I'm so scared of the exam results.",
    "I feel detached from everything. The stress has numbed me.",
    "I want to scream. The pressure from family and exams is too much.",
    "I can't remember anything I studied. My mind goes completely blank.",
    "I feel hopeless. No amount of studying seems to make me feel ready.",
    "I'm terrified. I've never felt this level of anxiety in my life.",
    "I feel like giving up entirely. The stress is physically painful.",
    "I avoid thinking about the exam because it triggers intense panic.",
]

# Augmentation phrases to add variety
PREFIXES = [
    "", "Honestly, ", "To be honest, ", "Well, ", "I think ", "I guess ",
    "Lately, ", "Recently, ", "Right now, ", "At this moment, ",
]

SUFFIXES = [
    "", " I just want it to be over.", " I hope things work out.",
    " I need to stay focused.", " Maybe I should talk to someone.",
    " I'm trying my best.", " I don't know what to do.",
]


def augment(text: str) -> str:
    """Apply random prefix/suffix augmentation."""
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    return f"{prefix}{text}{suffix}".strip()


def generate_dataset(samples_per_class: int = 500) -> str:
    """Generate the CSV dataset and return its path."""
    rows = []
    for _ in range(samples_per_class):
        rows.append((augment(random.choice(LOW_TEMPLATES)), "Low Anxiety"))
        rows.append((augment(random.choice(MODERATE_TEMPLATES)), "Moderate Anxiety"))
        rows.append((augment(random.choice(HIGH_TEMPLATES)), "High Anxiety"))

    random.shuffle(rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Dataset generated: {DATASET_PATH}  ({len(rows)} samples)")
    return DATASET_PATH


if __name__ == "__main__":
    generate_dataset()
