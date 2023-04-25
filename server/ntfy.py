import requests, json, time
from datetime import datetime



class ntfy:
    def __init__(self):
        self.topic = "a5_topic_4848rurirjfuurir"
        self.url = "https://ntfy.sh/"

    def notify(self, message):
        """Send a notification to the client.

        :param message: The message to send to the client.
        """

        # send a request to the nfty url with the message as a parameter

        title = "User in trouble"
        priority = "urgent"
        tags = "warning"

        user = "Harry"
        activity = "fallen"

        location_lat = "52.62177164741303"
        location_long = "1.2336131749025374"

        google_url = (
            "https://www.google.com/maps/search/?api=1&query="
            + location_lat
            + ","
            + location_long
        )

        message = (
            user
            + " has "
            + activity
            + """\nat """
            + str(datetime.now())
        )

        requests.post(
            self.url,
            data=json.dumps(
                {
                    "topic": self.topic,
                    "message": message,
                    "title": title,
                    "tags": ["warning"],
                    "priority": 5,
                    "actions": [
                        {"action": "view", "label": "Open Location", "url": google_url}
                    ],
                }
            ),
        )


# "Actions": "view, Open location, " + google_url + ", clear=true;",

notify = ntfy()
notify.notify("Hello")
