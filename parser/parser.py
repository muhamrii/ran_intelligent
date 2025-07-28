"""
Parse 3GPP XML/CSV RAN files, extract metadata into Pandas DataFrames.
"""
import pandas as pd
import xml.etree.ElementTree as ET

def parse_csv(file_path):
    # Dummy function for CSV parsing
    return pd.read_csv(file_path)

def parse_xml(file_path):
    """
    Parse the Ericsson CM LTE XML file and return:
      - A dictionary of DataFrames (dfs)
      - Detailed metadata (metadata)
      - Simplified metadata (metadata2)
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

    dfs = {table: pd.DataFrame(rows) for table, rows in dfs_data.items()}
    metadata2 = {table: list(details["parameters"].keys()) for table, details in metadata.items()}
    return dfs, metadata, metadata2