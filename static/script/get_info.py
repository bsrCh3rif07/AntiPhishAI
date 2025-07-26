import re
from urllib.parse import urlparse
import whois
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

def get_features_info(domain, email_content="", phishing_keywords=None, email_phishing_keywords=None):
    features = {}

    # Default phishing keywords if none are provided
    phishing_keywords = phishing_keywords or ['login', 'signin', 'secure', 'account', 'update', 'confirm', 'verify']
    email_phishing_keywords = email_phishing_keywords or ['urgent', 'confirm', 'banking', 'account', 'password', 'sensitive']

    # --- URL Features ---
    try:
        parsed_url = urlparse(domain)
        
        # Length of URL
        features['length_url'] = [len(domain)]
        
        # Length of hostname
        features['length_hostname'] = [len(parsed_url.hostname) if parsed_url.hostname else 0]
        
        # Check if domain is an IP address
        features['ip'] = [1 if re.match(r"\d+\.\d+\.\d+\.\d+", domain) else 0]
        
        # Count the number of dots in the domain
        features['nb_dots'] = [domain.count('.')]
        
        # Count the number of query parameters (?)
        features['nb_qm'] = [domain.count('?')]
        
        # Count the number of equals signs in the URL (used for parameters)
        features['nb_eq'] = [domain.count('=')]
        
        # Count slashes in the URL (path and structure)
        features['nb_slash'] = [domain.count('/')]
        
        # Count occurrences of "www" in the domain name
        features['nb_www'] = [domain.count('www')]

        # Prefix and suffix (check if the domain starts or ends with specific keywords like "login" or "secure")
        features['prefix_suffix'] = [1 if any(domain.startswith(keyword) for keyword in phishing_keywords) or any(domain.endswith(keyword) for keyword in phishing_keywords) else 0]

        # Shortest word in the hostname
        hostname_words = parsed_url.hostname.split('.')
        features['shortest_word_host'] = [min(len(word) for word in hostname_words) if hostname_words else 0]

        # Longest words in the URL (path, hostname, etc.)
        all_words = re.findall(r'\w+', domain)  # Find all words in the URL (hostname, path, etc.)
        features['longest_words_raw'] = [max(len(word) for word in all_words) if all_words else 0]
        
        # Ratio of digits in the URL
        features['ratio_digits_url'] = [sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0]
        
        # Ratio of digits in the hostname
        features['ratio_digits_host'] = [sum(c.isdigit() for c in parsed_url.hostname) / len(parsed_url.hostname) if parsed_url.hostname else 0]
        
        # Check if TLD is in subdomain
        features['tld_in_subdomain'] = [1 if parsed_url.hostname and domain.split('.')[0] in parsed_url.hostname else 0]
        
        # Longest word in the path
        path = parsed_url.path
        path_words = path.split('/')
        features['longest_word_path'] = [max(len(word) for word in path_words) if path_words else 0]
        
        # Phishing hints in the URL
        features['phish_hints'] = [1 if any(keyword in domain for keyword in phishing_keywords) else 0]
        
    except Exception as e:
        print(f"Error processing URL features: {e}")
        features = {key: [0] for key in [
            'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_qm', 'nb_eq',
            'nb_slash', 'nb_www', 'ratio_digits_url', 'ratio_digits_host',
            'tld_in_subdomain', 'longest_word_path', 'phish_hints'
        ]}

    # --- Count of Hyperlinks ---
    try:
        page = requests.get(domain, timeout=5, verify=False)
        soup = BeautifulSoup(page.content, 'html.parser')
        hyperlinks = soup.find_all('a')
        features['nb_hyperlinks'] = [len(hyperlinks)]
        
        # Ratio of internal hyperlinks
        internal_links = [link for link in hyperlinks if link.get('href', '').startswith(parsed_url.hostname)]
        features['ratio_intHyperlinks'] = [len(internal_links) / len(hyperlinks) if hyperlinks else 0]
    except Exception as e:
        print(f"Error processing hyperlinks: {e}")
        features['nb_hyperlinks'] = [0]
        features['ratio_intHyperlinks'] = [0]

    # --- Title-based Checks ---
    try:
        title = soup.title.string if soup.title else ""
        features['empty_title'] = [1 if not title else 0]
        features['domain_in_title'] = [1 if parsed_url.hostname and parsed_url.hostname.split('.')[0] in title else 0]
    except Exception as e:
        print(f"Error processing title-based checks: {e}")
        features['empty_title'] = [0]
        features['domain_in_title'] = [0]

    # --- Domain Age ---
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.get('creation_date')
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(creation_date, datetime):
            features['domain_age'] = [(datetime.now() - creation_date).days]
        else:
            features['domain_age'] = [0]
    except Exception as e:
        print(f"Error processing domain age: {e}")
        features['domain_age'] = [0]

    # --- Google Index and Page Rank (placeholders) ---
    features['google_index'] = [0]  # Replace with actual logic if available
    features['page_rank'] = [0]  # Placeholder

    # --- Email Content Features ---
    if email_content:
        try:
            features['email_phish_hints'] = [1 if any(keyword in email_content.lower() for keyword in email_phishing_keywords) else 0]
            features['generic_greeting'] = [1 if re.search(r'\b(hello|hi|dear)\b', email_content, re.IGNORECASE) else 0]
            features['click_here'] = [1 if 'click here' in email_content.lower() else 0]
            suspicious_link_patterns = [r'http[s]?://[^\s]+']
            features['suspicious_links'] = [1 if any(re.search(pattern, email_content) for pattern in suspicious_link_patterns) else 0]
            features['email_urgency'] = [1 if 'urgent' in email_content.lower() else 0]
            features['email_formatted'] = [1 if re.search(r'<.*?>', email_content) else 0]
        except Exception as e:
            print(f"Error processing email features: {e}")
            email_feature_keys = ['email_phish_hints', 'generic_greeting', 'click_here', 'suspicious_links', 'email_urgency', 'email_formatted']
            for key in email_feature_keys:
                features[key] = [0]

    return features
