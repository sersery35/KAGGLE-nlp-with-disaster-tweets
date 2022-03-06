import pandas as pd
import re
import os
import requests
import tweepy


def get_dataframe_from_csv(csv_file_name: str):
    """
    :param csv_file_name: str
    :return: pd.DataFrame()
    """
    if not csv_file_name.endswith(".csv"):
        raise FileNotFoundError('File must be a csv file.')

    file_dir = f'../data/{csv_file_name}'
    print(f"Getting the file: {file_dir}")
    return pd.read_csv(file_dir, sep=',')


def extract_urls(dataframe: pd.DataFrame):
    """
    method extracts urls from a dataframe
    :param dataframe: pd.DataFrame, dataframe to be processed
    :return: pd.DataFrame with an extra url column with the corresponding urls
    """
    # capture url domain and dir, join them to create a downloadable link then remove quotation marks
    urls = dataframe["text"].map(lambda text: ["".join([re.sub(r'[\"\']', "", element) for element in group])
                                               for group in re.findall(r'(https?:\/\/)(\S+?)(\/\S+)', text)])
    # for row in urls.values:
    #     if len(row) > 0:
    #         print(f"{row}")
    dataframe["url"] = urls
    return dataframe


def download_urls(dataframe):
    mentioned_tweets = []
    for row in dataframe["url"].values:
        mentioned_tweets_for_current_tweet = []
        if len(row) > 0:
            # download urls one by one
            for url in row:
                try:
                    response = requests.get(url)
                    print(response.content)
                except requests.exceptions.SSLError:
                    print(f"SSL cert of {url} has expired")

def main():
    train_file_name = 'train.csv'
    # test_file_name = 'test.csv'
    # sample_submission_file_name = 'sample_submission.csv'
    # max_vocabulary_size = 20000
    with open(os.path.join(os.getcwd(), '../bearer_token.txt'), 'r') as file:
        bearer_token = re.sub(r'\\n', "", file.read().strip())
    if bearer_token is None:
        raise ValueError('Please provide a bearer token in a file named bearer_token.txt in the root folder in '
                         'this project. ')
    df = get_dataframe_from_csv(train_file_name)

    df = extract_urls(df)
    sample = df.url[df.url[df.url.notna()].map(lambda x: len(x) > 0)].values[0]
    tweet_id = sample[0].split('/')[-1]
    print(f"Tweet id: {tweet_id}")
    # download_urls(df)
    # print(df)


if __name__ == '__main__':
    main()
