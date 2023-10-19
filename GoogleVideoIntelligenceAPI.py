from google.cloud import storage, videointelligence
from google.oauth2 import service_account
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# Path to your service account key file
key_path = "C:/Users/Asian/OneDrive/Desktop/AI/Research/Music Emotion Classification/turing-terminus-392618-aeb7c4cc7413.json"

# Create a credentials object
credentials = service_account.Credentials.from_service_account_file(key_path)

# Use the credentials when creating the clients
storage_client = storage.Client(credentials=credentials)
video_client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)

def analyze_videos_in_bucket(bucket_name):
    video_client = videointelligence.VideoIntelligenceServiceClient()

    features = [
        videointelligence.Feature.LABEL_DETECTION,
        videointelligence.Feature.SHOT_CHANGE_DETECTION,
        videointelligence.Feature.EXPLICIT_CONTENT_DETECTION
    ]

    # Get the list of objects in the bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()

    # Create empty lists to store the results
    label_results = []
    explicit_results = []
    shot_results = []

    for blob in blobs:
        if blob.name.endswith('.mp4'):
            input_uri = "gs://{}/{}".format(bucket_name, blob.name)
            operation = video_client.annotate_video(
                request={
                    "features": features,
                    "input_uri": input_uri,
                }
            )
            print("\nProcessing video {} for label annotations:".format(blob.name))

            result = operation.result(timeout=180)
            print("Finished processing video {}.".format(blob.name))

            # Label detection
            segment_labels = result.annotation_results[0].segment_label_annotations
            for segment_label in segment_labels:
                label_description = segment_label.entity.description
                for category_entity in segment_label.category_entities:
                    category_description = category_entity.description

                    for segment in segment_label.segments:
                        start_time = segment.segment.start_time_offset.seconds + segment.segment.start_time_offset.microseconds / 1e6
                        end_time = segment.segment.end_time_offset.seconds + segment.segment.end_time_offset.microseconds / 1e6
                        confidence = segment.confidence

                        # Append the results to the label results list
                        label_results.append({
                            "Video": blob.name,
                            "Label Description": label_description,
                            "Category Description": category_description,
                            "Start Time": start_time,
                            "End Time": end_time,
                            "Confidence": confidence
                        })

            # Explicit content detection
            explicit_content = result.annotation_results[0].explicit_annotation
            video_confidences = []  # List to store confidences for each video
            for frame in explicit_content.frames:
                time_offset = frame.time_offset.seconds + frame.time_offset.microseconds / 1e6
                pornography_likelihood = frame.pornography_likelihood

                # Append the results to the explicit content results list
                explicit_results.append({
                    "Video": blob.name,
                    "Label Description": "Explicit Content",
                    "Category Description": "N/A",
                    "Start Time": time_offset,
                    "End Time": time_offset,
                    "Confidence": pornography_likelihood
                })

                # Append the confidence to the video_confidences list
                video_confidences.append(pornography_likelihood)

            # Calculate the mean confidence for the video
            video_mean_confidence = sum(video_confidences) / len(video_confidences)

            # Append the mean confidence to the explicit content results for the video
            explicit_results.append({
                "Video": blob.name,
                "Label Description": "Mean Confidence",
                "Category Description": "N/A",
                "Start Time": None,
                "End Time": None,
                "Confidence": video_mean_confidence
            })

            # Shot change detection
            shot_annotations = result.annotation_results[0].shot_annotations
            for shot in shot_annotations:
                start_time = shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6
                end_time = shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6

                # Append the results to the shot detection results list
                shot_results.append({
                    "Video": blob.name,
                    "Label Description": "Shot Change",
                    "Category Description": "N/A",
                    "Start Time": start_time,
                    "End Time": end_time,
                    "Confidence": None  # No confidence value for shot change detection
                })

    # Create DataFrames from the results
    label_df = pd.DataFrame(label_results)
    explicit_df = pd.DataFrame(explicit_results)
    shot_df = pd.DataFrame(shot_results)

    # Create an Excel workbook
    workbook = Workbook()

    # Create sheets for each table
    label_sheet = workbook.active
    label_sheet.title = "Label Detection"
    explicit_sheet = workbook.create_sheet(title="Explicit Content Detection")
    shot_sheet = workbook.create_sheet(title="Shot Detection")

    # Write tables to the respective sheets
    for row in dataframe_to_rows(label_df, index=False, header=True):
        label_sheet.append(row)

    for row in dataframe_to_rows(explicit_df, index=False, header=True):
        explicit_sheet.append(row)

    for row in dataframe_to_rows(shot_df, index=False, header=True):
        shot_sheet.append(row)

    # Apply formatting to the tables
    header_font = Font(bold=True)
    alignment = Alignment(horizontal="center", vertical="center")

    for cell in label_sheet[1]:
        cell.font = header_font
        cell.alignment = alignment

    for cell in explicit_sheet[1]:
        cell.font = header_font
        cell.alignment = alignment

    for cell in shot_sheet[1]:
        cell.font = header_font
        cell.alignment = alignment

    # Save the Excel file
    excel_file = "video_results.xlsx"
    workbook.save(excel_file)
    print("Results saved to {}".format(excel_file))

# Specify your bucket name
bucket_name = "anime_food_landscape_object_bucket"

# Call the function to analyze videos in the bucket and save the results as an Excel file
analyze_videos_in_bucket(bucket_name)




# """Analyze Labels"""
# from google.cloud import videointelligence

# video_client = videointelligence.VideoIntelligenceServiceClient()
# features = [videointelligence.Feature.LABEL_DETECTION]
# operation = video_client.annotate_video(
#     request={
#         "features": features,
#         "input_uri": "gs://anime_food_landscape_object_bucket/Relaxing Anime Cooking ｜ Aesthetic Anime ASMR.mp4",
#     }
# )
# print("\nProcessing video for label annotations:")

# result = operation.result(timeout=180)
# print("\nFinished processing.")

# # first result is retrieved because a single video was processed
# segment_labels = result.annotation_results[0].segment_label_annotations
# for i, segment_label in enumerate(segment_labels):
#     print("Video label description: {}".format(segment_label.entity.description))
#     for category_entity in segment_label.category_entities:
#         print(
#             "\tLabel category description: {}".format(category_entity.description)
#         )

#     for i, segment in enumerate(segment_label.segments):
#         start_time = (
#             segment.segment.start_time_offset.seconds
#             + segment.segment.start_time_offset.microseconds / 1e6
#         )
#         end_time = (
#             segment.segment.end_time_offset.seconds
#             + segment.segment.end_time_offset.microseconds / 1e6
#         )
#         positions = "{}s to {}s".format(start_time, end_time)
#         confidence = segment.confidence
#         print("\tSegment {}: {}".format(i, positions))
#         print("\tConfidence: {}".format(confidence))
#     print("\n")
