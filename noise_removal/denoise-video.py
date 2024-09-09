import argparse

from ica import AutoSelector, GaussianSpatialFilter, VideoAnalyzer, VideoData


def main():
    parser = argparse.ArgumentParser(
        description="Denoise calcium image video with hotizontal line noise."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="path to the video to denoise",
    )
    parser.add_argument(
        "-s",
        "--gaussian-sigma",
        type=float,
        required=False,
        help='the "strength" of the gaussian filter. Default value 1.5',
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=False,
        help="path of the output denoised video. Default path out.avi",
    )
    parser.add_argument(
        "-n",
        "--save-not-selected-components",
        required=False,
        action="store_true",
        help="if in use, also save the video of the non selected components for debugging, saved as non_selected_{output_path}",
    )
    parser.add_argument(
        "--no-ffmpeg",
        required=False,
        action="store_true",
        help="""use if ffmpeg is not available in the system.
        It will save the video as mp4, and not lossless as would be with ffmpeg.
        Note that --output-path has to be an mp4 file.""",
    )

    args = parser.parse_args()

    # We put default values if not passed as arguments
    if args.output_path is None:
        if args.no_ffmpeg:
            output_path = "out.mp4"
        else:
            output_path = "out.avi"
    else:
        output_path = args.output_path

    if args.gaussian_sigma is None:
        sigma = 1.5
    else:
        sigma = args.gaussian_sigma

    # Initialize VideoProcessor with preprocessors and analysis methods
    video = VideoData(
        args.video_path,
    )

    # Load video
    video.load_video()

    analyzer = VideoAnalyzer(video, [GaussianSpatialFilter(sigma)])

    # Choose how many components to use
    analyzer.choose_n_pca_components_interactive()

    # Decompose video
    analyzer.decompose_video()

    # Select ICA components
    # Two options: component maps and variance images (more expensive)
    is_component_selected = AutoSelector().select_components(
        analyzer.get_component_maps()
    )
    print(f"Selected {sum(is_component_selected)} components")

    print("Saving denoised video")
    if args.no_ffmpeg:
        video.save_frames_as_mp4(
            analyzer.compose_video(is_component_selected=is_component_selected),
            output_path,
        )
    else:
        video.save_frames_lossless(
            analyzer.compose_video(is_component_selected=is_component_selected),
            output_path,
        )

    if args.save_not_selected_components:
        print("Saving video of not selected components")
        non_selected_components = [not selected for selected in is_component_selected]
        if args.no_ffmpeg:
            video.save_frames_as_mp4(
                analyzer.compose_video(is_component_selected=non_selected_components),
                f"non_selected_{output_path}",
            )
        else:
            video.save_frames_lossless(
                analyzer.compose_video(is_component_selected=non_selected_components),
                f"non_selected_{output_path}",
            )


if __name__ == "__main__":
    main()
