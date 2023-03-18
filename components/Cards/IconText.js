import React from 'react';
import Link from 'next/link';
import PropTypes from 'prop-types';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import { useText } from '/theme/common';
import useStyles from './cards-style';

export default function IconText(props) {
  const { classes, cx } = useStyles();
  const textStyle = useText();
  const {
    icon,
    text,
    desc,
    url,
  } = props;
  return (
    <Paper className={classes.iconText}>
      <div className={classes.icon}>
        <span className={icon} />
      </div>
      <Link href={`${url}`} target="_blank">
        <div className={classes.more}>
          <Typography className={cx(classes.name, textStyle.paragraph)} variant="h6" display="block">{text}</Typography>
          <Typography className={cx(classes.desc, textStyle.paragraph)} display="block">{desc}</Typography>
        </div>
      </Link>
    </Paper>
  );
}

IconText.propTypes = {
  icon: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
  desc: PropTypes.string.isRequired,
};
