import random
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak


def create_latin_square(n):
    return (np.arange(n)[:, None] + np.arange(n)) % n

def pseudo_latin_rectangle(m, n):
    latin_square = create_latin_square(n)
    repeated_rows = np.tile(latin_square, (m // n + 1, 1))[:m]
    np.random.shuffle(repeated_rows)
    return repeated_rows

def generate_study_matrix_models_latin_rectangle(rows, columns, shuffle_rows=True):
    len_columns = len(columns)

    latin_rec = pseudo_latin_rectangle(len(rows), len_columns).astype(object)
    columns_copy = columns.copy()
    if shuffle_rows:
        random.shuffle(columns_copy)

    rows_copy = rows.copy()
    random.shuffle(rows_copy)

    for i, outer in enumerate(latin_rec):
        row = rows_copy[i]
        for j, inner in enumerate(outer):
            column = columns_copy[int(inner)]
            latin_rec[i][j] = f"{row}\n{column}"

    latin_rec = latin_rec.tolist()
    return latin_rec

def generate_study_matrix_models_random(rows, columns):
    models_copy = rows.copy()
    options_copy = columns.copy()
    study_matrix = []
    random.shuffle(models_copy)
    for model in models_copy:
        random.shuffle(options_copy)
        inner_list = []
        for option in options_copy:
            inner_list.append(model + "\n" + str(option))
        study_matrix.append(inner_list)
    return study_matrix

def generate_evaluations(rows, columns, number_of_evaluations, eval_func):
    evaluations = []
    for _ in range(number_of_evaluations):
        evaluations.append(eval_func(rows, columns))
    return evaluations

def generate_complete_evaluation(evals):
    #format (title, rows, columns, number_of_evaluations, eval_func)
    evaluations = []
    for eval in evals:
        title, rows, columns, number_of_evaluations, eval_func = eval
        evaluations.append((title, generate_evaluations(rows, columns, number_of_evaluations, eval_func)))
    return evaluations

def generater_pdf_table_from_matrix(matrix, style):
    # Create a Table object with the matrix data
    table = Table(matrix)

    # Update the style to include a smaller font size
    style.extend([
        ('FONTSIZE', (0, 0), (-1, -1), 5),  # Set the font size to 8
        ('LEFTPADDING', (0, 0), (-1, -1), 3),  # Reduce left padding
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),  # Reduce right padding
    ])

    # Apply the style to the table
    table.setStyle(TableStyle(style))

    return table

def save_evalauation_to_pdf(evaluation, filename):
    # Document setup
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()

    for eval in evaluation:
        title, matrices = eval
        for i, matrix in enumerate(matrices):
            sub_title = title + " " + str(i+1)
            elements.append(Paragraph(sub_title, styles['Heading4']))
            # Define the style for the table here, e.g., borders, alignment, etc.
            style = [('GRID', (0, 0), (-1, -1), 1, colors.black)]
            table = generater_pdf_table_from_matrix(matrix, style)
            elements.append(table)
        # new page for each evaluation
        elements.append(PageBreak())


    # Build the document with the elements
    doc.build(elements)


def main():
    to_eval = []
    models1 = ['audioldm-s-full-v2', 'audioldm-m-full', 'audioldm-l-full', 'audioldm2', 'audioldm2-large', 'audioldm2-music']
    devices = ['cuda', 'mps', 'cpu']
    to_eval.append(("device_evaluation", models1, devices, 5, generate_study_matrix_models_latin_rectangle))


    guidance_scales = [1, 2, 3, 4, 5]
    to_eval.append(("guidance_scale_evaluation", models1, guidance_scales, 3, generate_study_matrix_models_latin_rectangle))


    inference_steps = [5, 10, 20, 50, 100, 200, 400]
    to_eval.append(("inference_steps_evaluation", models1, inference_steps, 3, generate_study_matrix_models_latin_rectangle))


    length = [5, 10, 20, 30]
    models2 = ['audioldm-m-full', 'audioldm2']
    to_eval.append(("length_evaluation", models2, length, 3, generate_study_matrix_models_latin_rectangle))


    negative_prompts = ['low quality', 'average quality', 'harsh noise', 'dissonant chords', 'distorted sounds', 'clashing frequencies', 'feedback loop', 'clattering', 'inharmonious', 'average quality', 'noise', 'high pitch', 'artefacts']
    models3 = ['audioldm-m-full', 'audioldm2-music']
    to_eval.append(("negative_prompts_evaluation", models3, negative_prompts, 3, generate_study_matrix_models_random))


    models4 = ['audioldm-m-full', 'audioldm-l-full', 'audioldm2', 'audioldm2-music']
    prompts_single_events = ['A kickdrum', 'A single kickdrum', 'A snare', 'A single snare', 'A single Light triangle ting', 'Loud clap sound', 'A gong hit']
    to_eval.append(("single_events_evaluation", prompts_single_events, models4, 1, generate_study_matrix_models_latin_rectangle))

    prompts_instruments = ['FM synthesis bells', 'mellotron chords', 'A bagpipe melody', 'A guitar string', 'A piano chord']
    to_eval.append(("instruments_evaluation", prompts_instruments, models4, 1, generate_study_matrix_models_latin_rectangle))

    prompts_adjectives = ['Dark pad sound', 'An ethereal shimmering synth pad', 'An angelic choir', 'dreamy nostalgic strings', 'a sad violin solo']
    to_eval.append(("adjectives_evaluation", prompts_adjectives, models4,  1, generate_study_matrix_models_latin_rectangle))

    prompts_effect = ['Long sustain snare hit', 'A fluttering harp with crystal echoes', 'A Synth with a delay effect', 'echoing synth stabs', 'A distorted synth', 'A detuned synth', 'Reverse cymbal', 'A kickdrum with a lot of reverb']
    to_eval.append(("effect_evaluation",prompts_effect, models4, 1, generate_study_matrix_models_latin_rectangle))

    prompts_music_production = ['an 808 kickdrum', 'the amen break', 'a 909 snare', 'a 303 baseline', 'A jungle drum break', 'A Juno-106 pad', 'Oberheimer OB-Xa string pads']
    to_eval.append(("music_production_evaluation", prompts_music_production, models4, 1, generate_study_matrix_models_latin_rectangle))

    eval = generate_complete_evaluation(to_eval)
    print(eval[0][1][0])

    # Save the evaluation matrix to a PDF
    save_evalauation_to_pdf(eval, "evaluation.pdf")


if __name__ == "__main__":
    main()
