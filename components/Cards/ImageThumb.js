import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';
import Paper from '@mui/material/Paper';
import { useText } from '/theme/common';
import useStyles from './cards-style';

export default function ImageThumb(props) {
  const { classes, cx } = useStyles();
  const { classes: text } = useText();
  const {
    img,
    title,
    link,
    size,
  } = props;

  const setSize = sizePaper => {
    switch (sizePaper) {
      case 'short':
        return classes.short;
      case 'long':
        return classes.long;
      default:
        return classes.medium;
    }
  };
  return (
    <Paper className={cx(classes.imgThumb, setSize(size))}>
      <div className={classes.figure}>
        <div className={classes.img} style={{ backgroundImage: `url(${img})` }} />
      </div>
      <div className={classes.detail}>
        <Typography variant="h6" className={text.subtitle}>{title}</Typography>
        <Link href={link} target="_blank">{link}</Link>
      </div>
    </Paper>
  );
}

ImageThumb.propTypes = {
  img: PropTypes.string.isRequired,
  title: PropTypes.string.isRequired,
  link: PropTypes.string.isRequired,
  size: PropTypes.string.isRequired,
};
