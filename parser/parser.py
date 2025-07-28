"""
Parse 3GPP XML/CSV RAN files, extract metadata into Pandas DataFrames.
"""
import pandas as pd
import xml.etree.ElementTree as ET
import nlpaug.augmenter.word as naw
from nlpaug.augmenter.word import ContextualWordEmbsAug


def parse_csv(file_path):
    # Dummy function for CSV parsing
    return pd.read_csv(file_path)

def parse_xml(file_path):
    """
    Parse the Ericsson CM LTE XML file and return:
      - A dictionary of DataFrames (dfs)
      - Detailed metadata (metadata)
      - Simplified metadata (metadata2)
      - NER dataset (ner_dataset)
    """
    def clean_tag(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    def remove_vsdata_prefix(s):
        exceptions = {"vsDataType", "vsDataFormatVersion"}
        if s.startswith("vsData") and s not in exceptions:
            return s[len("vsData"):]
        return s

    def clean_key(key):
        if "." in key:
            parts = key.split(".", 1)
            parts[0] = remove_vsdata_prefix(parts[0])
            return parts[0] + "." + parts[1]
        else:
            return remove_vsdata_prefix(key)

    def update_metadata(table, row):
        if table not in metadata:
            metadata[table] = {"parameters": {}}
        for key in row:
            new_key = clean_key(key)
            if new_key not in metadata[table]["parameters"]:
                metadata[table]["parameters"][new_key] = "No description available"

    tree = ET.parse(file_path)
    root = tree.getroot()

    dfs_data = {}
    metadata = {}
    ner_dataset = []

    entity_to_label = {
        "B-TABLE": 1,
        "B-COLUMN": 2,
        "B-VALUE": 3,
        "O": 0  # Outside any entity
    }

    # Extract global date from fileFooter (if present)
    date = ''
    file_footer = root.find('{configData.xsd}fileFooter')
    if file_footer is not None:
        date = file_footer.attrib.get('dateTime', '')

    for config in root.findall('{configData.xsd}configData'):
        for child1 in config:
            for subnetwork in child1.findall('{genericNrm.xsd}SubNetwork'):
                area_name = subnetwork.attrib.get('id', '')
                for mecontext in subnetwork.findall('{genericNrm.xsd}MeContext'):
                    cell_id = mecontext.attrib.get('id', '')
                    base_info = {'dateTime': date, 'Area_Name': area_name, 'CellId': cell_id}

                    # Process VsDataContainer directly under MeContext
                    for vs_container in mecontext.findall('{genericNrm.xsd}VsDataContainer'):
                        row = base_info.copy()
                        table_name = None
                        for elem in vs_container.iter():
                            tag_cleaned = clean_tag(elem.tag)
                            if tag_cleaned == 'vsDataType' and elem.text:
                                table_name = remove_vsdata_prefix(elem.text.strip())
                            if elem.text and elem.text.strip():
                                row[clean_key(tag_cleaned)] = elem.text.strip()
                        if table_name is None:
                            table_name = 'Unknown'
                        update_metadata(table_name, row)
                        dfs_data.setdefault(table_name, []).append(row)

                    # Process ManagedElement nodes under MeContext (e.g. EnodeB info)
                    for managed_element in mecontext.findall('{genericNrm.xsd}ManagedElement'):
                        id2 = managed_element.attrib.get('id', '')
                        enodeb_info = base_info.copy()
                        enodeb_info['Id2'] = id2
                        attributes = managed_element.find('{genericNrm.xsd}attributes')
                        if attributes is not None:
                            for attr in attributes:
                                key = clean_key(clean_tag(attr.tag))
                                enodeb_info[key] = attr.text.strip() if attr.text else ''
                        update_metadata('EnodeBInfo', enodeb_info)
                        dfs_data.setdefault('EnodeBInfo', []).append(enodeb_info)

                        # Process nested VsDataContainer (one level deep)
                        for vs_container in managed_element.findall('{genericNrm.xsd}VsDataContainer'):
                            for nested_vs in vs_container.findall('{genericNrm.xsd}VsDataContainer'):
                                id3 = nested_vs.attrib.get('id', '')
                                row = base_info.copy()
                                row['Id2'] = id2
                                row['Id3'] = id3
                                table_name = None
                                attributes = nested_vs.find('{genericNrm.xsd}attributes')
                                if attributes is not None:
                                    for attr in attributes:
                                        tag_cleaned = clean_tag(attr.tag)
                                        if tag_cleaned == 'vsDataType' and attr.text:
                                            table_name = remove_vsdata_prefix(attr.text.strip())
                                        if list(attr):
                                            parent = remove_vsdata_prefix(tag_cleaned)
                                            for sub in attr:
                                                sub_tag = clean_tag(sub.tag)
                                                key = f"{parent}.{sub_tag}"
                                                row[clean_key(key)] = sub.text.strip() if sub.text else ''
                                        else:
                                            key = clean_key(tag_cleaned)
                                            row[key] = attr.text.strip() if attr.text else ''
                                if table_name is None:
                                    table_name = 'Unknown'
                                update_metadata(table_name, row)
                                dfs_data.setdefault(table_name, []).append(row)

                        # Process deeper nested VsDataContainer (two levels deep)
                        for vs_container in managed_element.findall('{genericNrm.xsd}VsDataContainer'):
                            for nested_vs in vs_container.findall('{genericNrm.xsd}VsDataContainer'):
                                for deeper_vs in nested_vs.findall('{genericNrm.xsd}VsDataContainer'):
                                    id3 = nested_vs.attrib.get('id', '')
                                    id4 = deeper_vs.attrib.get('id', '')
                                    row = base_info.copy()
                                    row['Id2'] = id2
                                    row['Id3'] = id3
                                    row['Id4'] = id4
                                    table_name = None
                                    attributes = deeper_vs.find('{genericNrm.xsd}attributes')
                                    if attributes is not None:
                                        for attr in attributes:
                                            tag_cleaned = clean_tag(attr.tag)
                                            if tag_cleaned == 'vsDataType' and attr.text:
                                                table_name = remove_vsdata_prefix(attr.text.strip())
                                            if list(attr):
                                                parent = remove_vsdata_prefix(tag_cleaned)
                                                for sub in attr:
                                                    sub_tag = clean_tag(sub.tag)
                                                    key = f"{parent}.{sub_tag}"
                                                    row[clean_key(key)] = sub.text.strip() if sub.text else ''
                                            else:
                                                key = clean_key(tag_cleaned)
                                                row[key] = attr.text.strip() if attr.text else ''
                                    if table_name is None:
                                        table_name = 'Unknown'
                                    update_metadata(table_name, row)
                                    dfs_data.setdefault(table_name, []).append(row)

    # Create NER dataset
    for table_name, rows in dfs_data.items():
        for row in rows:
            # Ensure the row is a dictionary
            if not isinstance(row, dict):
                print(f"Skipping invalid row in table '{table_name}': {row}")
                continue

            tokens = []
            ner_tags = []

            # Add table name as a token
            tokens.append(table_name)
            ner_tags.append(entity_to_label["B-TABLE"])

            # Add columns and values as tokens
            for column, value in row.items():
                tokens.append(column)
                ner_tags.append(entity_to_label["B-COLUMN"])
                tokens.append(value)
                ner_tags.append(entity_to_label["B-VALUE"])

            # Append the sample to the dataset
            ner_dataset.append({"tokens": tokens, "ner_tags": ner_tags})

    # Calculate the required number of O and Time-Related tokens
    b_column_count = sum(1 for sample in ner_dataset for tag in sample["ner_tags"] if tag == entity_to_label["B-COLUMN"])
    required_count = int(b_column_count * 1.1)  # 110% of B-COLUMN count

    # Add variative 'O' tokens using NLPAug
    aug = ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    o_token_count = 0
    while o_token_count < required_count:
        try:
            # Generate a variative sentence
            augmented_output = aug.augment("This is a random sentence.")
            
            # Ensure the output is a string
            if isinstance(augmented_output, list):
                sentence = " ".join(augmented_output)  # Join list into a single string
            else:
                sentence = augmented_output
            
            tokens = sentence.split()
            
            # Ensure the tokens are valid and non-empty
            if tokens:
                ner_tags = [entity_to_label["O"]] * len(tokens)
                ner_dataset.append({"tokens": tokens, "ner_tags": ner_tags})
                o_token_count += len(tokens)
        except Exception as e:
            print(f"Error during augmentation: {e}")
            break

    # Ensure the total count of O tokens is logged
    print(f"Total O tokens added: {o_token_count}")

    # Add time-related tokens (new classified NER)
    time_related_tokens = [
        {"tokens": ["2025-07-28", "12:00", "PM"], "ner_tags": [4, 4, 4]},
        {"tokens": ["July", "28th", "2025"], "ner_tags": [4, 4, 4]},
        {"tokens": ["today"], "ner_tags": [4]},
        {"tokens": ["tomorrow"], "ner_tags": [4]},
        {"tokens": ["yesterday"], "ner_tags": [4]},
        {"tokens": ["next", "week"], "ner_tags": [4, 4]},
        {"tokens": ["last", "month"], "ner_tags": [4, 4]},
        {"tokens": ["in", "two", "days"], "ner_tags": [4, 4, 4]},
        {"tokens": ["three", "hours", "ago"], "ner_tags": [4, 4, 4]},
        {"tokens": ["next", "Monday"], "ner_tags": [4, 4]},
        {"tokens": ["this", "morning"], "ner_tags": [4, 4]},
        {"tokens": ["tonight"], "ner_tags": [4]},
        {"tokens": ["at", "noon"], "ner_tags": [4, 4]},
        {"tokens": ["midnight"], "ner_tags": [4]},
    ]

    # Add all months
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for month in months:
        time_related_tokens.append({"tokens": [month], "ner_tags": [4]})

    # Add all days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in days:
        time_related_tokens.append({"tokens": [day], "ner_tags": [4]})

    # Add quarters
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    for quarter in quarters:
        time_related_tokens.append({"tokens": [quarter], "ner_tags": [4]})

    time_related_count = sum(len(sample["tokens"]) for sample in time_related_tokens)

    # Add more time-related tokens if needed to meet the required count
    while o_token_count + time_related_count < required_count:
        time_related_tokens.append({"tokens": ["2025", "07", "28"], "ner_tags": [4, 4, 4]})
        time_related_count += 3

    ner_dataset.extend(time_related_tokens)

    dfs = {table: pd.DataFrame(rows) for table, rows in dfs_data.items()}
    metadata2 = {table: list(details["parameters"].keys()) for table, details in metadata.items()}
    return dfs, metadata, metadata2, ner_dataset