from sinf.utils import jobs
from sinf.data import fetal
from sinf.experiments import analysis
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", help="subject id of the experiment")
    parser.add_argument("--job_id", help="job name of the experiment")
    args = parser.parse_args()
    subject_id = args.subject_id
    job_id = args.job_id
    kwargs = jobs.get_job_args(job_id)
    nf = jobs.load_model_for_job(job_id)
    video, affine = fetal.get_video_for_subject(subject_id, 'even')
    analysis.evaluate_metrics_3d(nf, video.cuda(), out_dir=kwargs['paths']["job output dir"])


if __name__ == "__main__":
    main()